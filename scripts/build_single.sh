echo "=======Configuring and building ORB_SLAM2======= ..."
cd ..
rm -rf build
cmake -H. -Bbuild -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j12