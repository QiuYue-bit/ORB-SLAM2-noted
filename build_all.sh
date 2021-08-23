echo "Configuring and building Thirdparty/DBoW2 ..."
cd Thirdparty/DBoW2
rm -rf build
cmake -H. -Bbuild -DCMAKE_BUILD_TYPE=Release
cmake --build build -j12


pwd
cd ../g2o
pwd
echo "Configuring and building Thirdparty/g2o ..."
rm -rf build
cmake -H. -Bbuild -DCMAKE_BUILD_TYPE=Release
cmake --build build -j12

cd ../../

echo "Uncompress vocabulary ..."
cd Vocabulary
tar -xf ORBvoc.txt.tar.gz
cd ..

echo "Configuring and building ORB_SLAM2 ..."

rm -rf build
cmake -H. -Bbuild -DCMAKE_BUILD_TYPE=Release
cmake --build build -j12
