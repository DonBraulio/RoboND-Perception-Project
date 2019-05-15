#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml

object_list_param = None
dropboxes = None
test_scene = None

# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Voxel Grid filter
def voxel_downsample(cloud):
    vox = cloud.make_voxel_grid_filter()
    LEAF_SIZE = 0.01
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    return vox.filter()

# PassThrough filter
def passthrough_filter(cloud):
    passthrough = cloud.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.5
    axis_max = 1.1 
    passthrough.set_filter_limits(axis_min, axis_max)
    return passthrough.filter()


# RANSAC plane segmentation
def ransac_segment(cloud):
    seg = cloud.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.03
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()
    objects = cloud.extract(inliers, negative=True)
    table = cloud.extract(inliers, negative=False)
    return objects, table


# Extract outliers
def statistical_outliers_extract(cloud):
    fil = cloud.make_statistical_outlier_filter()
    fil.set_mean_k(50)
    fil.set_std_dev_mul_thresh(0.5)
    return fil.filter()


def kmean_get_clusters(cloud):
    xyz_cloud = XYZRGB_to_XYZ(cloud)
    tree = xyz_cloud.make_kdtree()
    ec = xyz_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold 
    ec.set_ClusterTolerance(0.05)
    ec.set_MinClusterSize(100)
    ec.set_MaxClusterSize(2000)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    # Assign colors to points from each cluster
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append(
                                    [xyz_cloud[indice][0],
                                     xyz_cloud[indice][1],
                                     xyz_cloud[indice][2],
                                     rgb_to_float(cluster_color[j])])
    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
    return cluster_indices, cluster_cloud


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    # Convert ROS msg to PCL data
    pcl_cloud = ros_to_pcl(pcl_msg)

    cloud = voxel_downsample(pcl_cloud)
    cloud= statistical_outliers_extract(cloud)
    cloud = passthrough_filter(cloud)

    pcl_filtered_pub.publish(pcl_to_ros(cloud))

    # segment objects/table
    cloud_objects, cloud_table = ransac_segment(cloud)

    # # Clusterize and assign colors
    cluster_indices, cluster_cloud = kmean_get_clusters(cloud_objects)

    # Generate and publish ROS messages
    pcl_objects_pub.publish(pcl_to_ros(cloud_objects))
    pcl_table_pub.publish(pcl_to_ros(cloud_table))
    pcl_cluster_pub.publish(pcl_to_ros(cluster_cloud))

    # Classify the clusters!
    detected_objects_labels = []
    detected_objects = []
    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster = cloud_objects.extract(pts_list)

        # Extract histogram features
        ros_cluster = pcl_to_ros(pcl_cluster)
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(cloud_objects[pts_list[0]])[0:3]
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    # This is the output you'll need to complete the upcoming project!
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # This message is common to all objects
    test_scene_num = Int32()
    test_scene_num.data = test_scene

    # list to keep all ROS messages in yaml format
    ros_messages = []

    # Loop through the pick list
    for o in object_list:

        # Pick pose = object centroid
        pick_pose = Pose()
        points_arr = ros_to_pcl(o.cloud).to_array()
        object_centroid = np.asscalar(np.mean(points_arr, axis=0)[:3])
        pick_pose.x, pick_pose.y, pick_pose.z = object_centroid

        # Place pose (corresponding dropbox position)
        place_pose = Pose()
        object_group = object_list_param[o.label]
        dropbox_position = dropboxes[object_group]['position']
        place_pose.x, place_pose.y, place_pose.z = dropbox_position

        # Other ROS message fields
        object_name = String()
        object_name.data = o.label
        arm_name = String()
        arm_name.data = dropboxes[object_group]['name']

        # Create dict and append to list in yaml format
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        ros_messages.append(yaml_dict)

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # Insert your message variables to be sent as a service request
            resp = pick_place_routine(test_scene, o.label, arm_name, centroid, dropbox_position)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # Save yaml
    send_to_yaml('output.yaml', ros_messages)


if __name__ == '__main__':
    rospy.init_node('perception', anonymous=True)

    # pick_list_x.yaml->dict {name: group}
    object_list_param = {d['name']: d['group'] for d in rospy.get_param('/object_list')}
    # dropbox.yaml->dict {group: {name, position: [x, y, z], group}}
    dropboxes = {d['group']: d for d in rospy.get_param('/dropbox')}

    # subscribe to point cloud
    pcl_sub = rospy.Subscriber("/pr2/world/points",
                               pc2.PointCloud2,
                               pcl_callback, queue_size=1)

    # publishers
    pcl_filtered_pub = rospy.Publisher("/pcl_filtered", PointCloud2, queue_size=1)
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # Select which trained model to use according to object list
    n_objects = len(object_list_param)
    if n_objects == 3:
        model_file = 'model_1.sav'
        test_scene = 1
    elif n_objects == 4:
        model_file = 'model_2.sav'
        test_scene = 2
    else:
        model_file = 'model_3.sav'
        test_scene = 3

    # Load Model From disk
    model = pickle.load(open(model_file, 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # main loop
    while not rospy.is_shutdown():
        rospy.spin()
