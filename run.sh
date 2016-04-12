echo "Starting..."
echo
echo "Stabilizing videos in dataset"

sh pre_processing.sh ~/png ~/cvpr2016codes/png_stab/ ~/cvpr2016codes/frame_lists/

echo
echo "Create training and testing windows from frame lists"
mkdir windows_frame_lists

cd preprocessing
g++ prepare_sliding_windows_from_frame_lists.cpp -o prepare_sliding_windows_from_frame_lists
cd ..

./preprocessing/prepare_sliding_windows_from_frame_lists train_videos.txt /home/suriya/cvpr2016codes/datasets/gtea/labels/ /home/suriya/cvpr2016codes/frame_lists/ /home/suriya/cvpr2016codes/windows_frame_lists/ /home/suriya/cvpr2016codes/png_stab/ train_windows.txt
./preprocessing/prepare_sliding_windows_from_frame_lists test_videos.txt /home/suriya/cvpr2016codes/datasets/gtea/labels/ /home/suriya/cvpr2016codes/frame_lists/ /home/suriya/cvpr2016codes/windows_frame_lists/ /home/suriya/cvpr2016codes/png_stab/ test_windows.txt

echo
echo "Preprocessing Finish"

echo
echo "Extract TDD features.."

mkdir tdd_features
mkdir idt_features

cd ego_tdd;
/usr/local/matlab2013b//bin/matlab -nodisplay -r 'gtea, exit()' 2>tdd_log.txt