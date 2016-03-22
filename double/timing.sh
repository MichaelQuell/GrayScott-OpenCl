let old=8
let new=16
for i in {0..6}
do
let old+=$old
#echo $old
let new+=$new
#echo $new
sed -i "s/$old/$new/" ./INPUTFILE
time ./grayscottOpenCLs
done
sed -i "s/$new/32/" ./INPUTFILE


