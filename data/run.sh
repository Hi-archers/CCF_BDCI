set -eux

cd /data/code
echo "Begin data preprocessing..."
python process_data.py
sleep 3

python split_yearmonday.py
python split_yearmonday_val.py
echo "Finish data preprocessing..."
sleep 3

echo "Begin bert inference..."
python bert_inference.py
echo "Finish bert inference..."


echo "Begin T5 inference..."
python inference.py && python inference_day.py && python inference_local.py && python inference_mon.py && python inference_org.py && python inference_person.py && python inference_year.py && python inference_yearday.py
echo "Finish T5 inference..."
sleep 3

python concat.py
echo "Finish generate result..."

