cmake_minimum_required(VERSION 3.18)

project(YOLOv11TRT LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_STANDALONE "Build standalone executable (main.cpp)" ON)
option(BUILD_ROS2      "Build ROS2 perception node"           OFF)

set(TENSORRT_INCLUDE_DIR "/usr/include/x86_64-linux-gnu")
set(TENSORRT_LIB_DIR "/usr/lib/x86_64-linux-gnu")
set(TENSORRT_INCLUDE_DIR ${TENSORRT_INCLUDE_DIR} CACHE PATH "TensorRT include path")
set(TENSORRT_LIB_DIR ${TENSORRT_LIB_DIR} CACHE PATH "TensorRT lib path")

find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV or set OpenCV_DIR.")
endif()

find_package(CUDA REQUIRED)
if(NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA not found. Please install the CUDA Toolkit.")
endif()

if(WIN32)
    set(TENSORRT_LIBS
        "${TENSORRT_LIB_DIR}/nvinfer.lib"
        "${TENSORRT_LIB_DIR}/nvonnxparser.lib"
        "${TENSORRT_LIB_DIR}/nvparsers.lib"
        "${TENSORRT_LIB_DIR}/nvinfer_plugin.lib"
    )
else()
    set(TENSORRT_LIBS
        nvinfer
        nvonnxparser
        nvinfer_plugin
    )
endif()

set(CORE_SOURCES
    src/yolov11.cpp
    src/preprocess.cu
)

set(CORE_HEADERS
    include/yolov11.h
    include/preprocess.h
    include/common.h
    include/cuda_utils.h
    include/logging.h
    include/macros.h
)

add_library(yolov11trt SHARED ${CORE_SOURCES} ${CORE_HEADERS})

target_compile_definitions(yolov11trt PRIVATE API_EXPORTS)

target_include_directories(yolov11trt PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/yolov11trt>
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src/
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
)

target_link_libraries(yolov11trt PRIVATE
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBS}
)

set_target_properties(yolov11trt PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    VERSION 1.0.0
    SOVERSION 1
)

install(TARGETS yolov11trt
    LIBRARY DESTINATION lib
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION include/yolov11trt
    FILES_MATCHING PATTERN "*.h"
)

if(BUILD_STANDALONE)
    add_executable(${PROJECT_NAME} main.cpp)
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIR}
    )
    target_link_libraries(${PROJECT_NAME} PRIVATE
        yolov11trt
        ${OpenCV_LIBS}
    )
endif()

if(BUILD_ROS2)
    find_package(rclcpp REQUIRED)
    find_package(cv_bridge REQUIRED)
    find_package(sensor_msgs REQUIRED)
    find_package(vision_msgs REQUIRED)
    find_package(CURL REQUIRED)

    add_executable(yolov11_ros2_node ros2/perception_node.cpp)
    target_link_libraries(yolov11_ros2_node PRIVATE
        yolov11trt
        rclcpp::rclcpp
        cv_bridge::cv_bridge
        vision_msgs::vision_msgs__rosidl_generator_cpp
        vision_msgs::vision_msgs__rosidl_typesupport_cpp
        sensor_msgs::sensor_msgs__rosidl_generator_cpp
        sensor_msgs::sensor_msgs__rosidl_typesupport_cpp
        CURL::libcurl
    )
    target_include_directories(yolov11_ros2_node PRIVATE
        ${CUDA_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIR}
        /opt/ros/humble/include/vision_msgs
    )

    install(TARGETS yolov11_ros2_node
        DESTINATION lib/${PROJECT_NAME}
    )

    add_executable(yolov11_visualization_node ros2/visualization_node.cpp)
    target_link_libraries(yolov11_visualization_node PRIVATE
        rclcpp::rclcpp
        cv_bridge::cv_bridge
        sensor_msgs::sensor_msgs__rosidl_generator_cpp
        sensor_msgs::sensor_msgs__rosidl_typesupport_cpp
        vision_msgs::vision_msgs__rosidl_generator_cpp
        vision_msgs::vision_msgs__rosidl_typesupport_cpp
        CURL::libcurl
    )
    target_include_directories(yolov11_visualization_node PRIVATE
        /opt/ros/humble/include/vision_msgs
    )

    install(TARGETS yolov11_visualization_node
        DESTINATION lib/${PROJECT_NAME}
    )

    install(DIRECTORY ros2/launch/
        DESTINATION share/${PROJECT_NAME}/launch
        FILES_MATCHING PATTERN "*.py"
    )
endif()