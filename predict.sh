set -e

input=$1

baseinput=${input##*/}
mkdir -p "tmp/segments-$baseinput"
mkdir -p "tmp/npy-$baseinput"
ffmpeg -loglevel error -i "$input" -f segment -segment_time 1 -c pcm_mulaw -vn "tmp/segments-$baseinput/$baseinput-%05d.wav"
./preprocess_audio_segments.py "tmp/segments-$baseinput" "tmp/npy-$baseinput"
./run_classifier.py "tmp/npy-$baseinput" model.json weights.hdf5 output.json
#rm -rf "tmp/segments-$baseinput"
#rm -rf "tmp/npy-$baseinput"
