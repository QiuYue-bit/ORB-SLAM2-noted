echo "=======Configuring and building Thirdparty/DBoW2======= ..."

rm -rf build
cmake -H. -Bbuild -DCMAKE_BUILD_TYPE=Release
cmake --build build -j12