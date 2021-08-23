echo "=======Configuring and building Thirdparty/g2o======= ..."

rm -rf build
cmake -H. -Bbuild -DCMAKE_BUILD_TYPE=Release
cmake --build build -j12