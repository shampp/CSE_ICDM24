cd K4/
echo "K4"
python3 pruning_exp.py
cd ../K8/
echo "K8"
python3 pruning_exp.py
echo "K16"
cd ../K16/
python3 pruning_exp.py
echo "K32"
cd ../K32/
python3 pruning_exp.py
echo "K64"
cd ../K64/
python3 pruning_exp.py


cd ../K16_1/
echo "K16_1"
python3 pruning_exp.py
cd ../K16_1p5/
echo "K16_1p5"
python3 pruning_exp.py
cd ../K16_2/
echo "K16_2"
python3 pruning_exp.py
cd ../K16_2p5/
echo "K16_2p5"
python3 pruning_exp.py
cd ../K16_3/
echo "K16_3"
python3 pruning_exp.py


cd ../N5/
echo "N5"
python3 pruning_exp.py
cd ../N10/
echo "N10"
python3 pruning_exp.py
echo "N25"
cd ../N25/
python3 pruning_exp.py
echo "N50"
cd ../N50/
python3 pruning_exp.py
echo "N100"
cd ../N100/
python3 pruning_exp.py


