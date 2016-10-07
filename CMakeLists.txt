cmake_minimum_required(VERSION 3.4)

project(TestDNN)

# Find packages.
find_package(dlib REQUIRED)

if (NOT dlib_FOUND)
  message(FATAL_ERROR "dlib is not found.")
endif()

find_package(OpenCV REQUIRED core highgui imgcodecs)

if(NOT OpenCV_FOUND)
  message(FATAL_ERROR "Open CV is not found.")
endif()

option(USE_CUDA "Use CUDA and cuDNN." OFF)
if (${USE_CUDA})
  find_package(CUDA)
  if(NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA is not found.")
  endif()
endif()

# End find packages.

set(MNIST_SOURCES ${CMAKE_CURRENT_LIST_DIR}/ReadMNIST/ReadMNISTDatas.cpp ${CMAKE_CURRENT_LIST_DIR}/ReadMNIST/ReadMNISTDatas.hpp)

configure_file(${CMAKE_CURRENT_LIST_DIR}/TrainFilePath.hpp.in ${CMAKE_BINARY_DIR}/TrainFilePath.hpp)

include_directories(${OpenCV_INCLUDE_DIRS} ${dlib_INCLUDE_DIRS} ${CMAKE_BINARY_DIR})

add_executable(ConvertMNISTLabelAndImage 
  ${CMAKE_CURRENT_LIST_DIR}/ReadMNIST/ConvertLabelAndImage.cpp
  ${MNIST_SOURCES}
  )

include_directories(${CMAKE_CURRENT_LIST_DIR}/ReadMNIST/)

if (USE_CUDA)
  cuda_add_executable(TestDNN 
    ${MNIST_SOURCES}
    ${CMAKE_CURRENT_LIST_DIR}/TestDNN/TestDNN.cpp
    ${CMAKE_CURRENT_LIST_DIR}/TestDNN/DLIBInput.hpp
    )
else()
  add_executable(TestDNN 
    ${MNIST_SOURCES}
    ${CMAKE_CURRENT_LIST_DIR}/TestDNN/TestDNN.cpp
    ${CMAKE_CURRENT_LIST_DIR}/TestDNN/DLIBInput.hpp
    )
endif()

target_link_libraries(ConvertMNISTLabelAndImage ${OpenCV_LIBRARIES})
target_link_libraries(TestDNN ${OpenCV_LIBRARIES} ${dlib_LIBRARIES})