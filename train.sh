python -u train.py --config models/config/VIGOR/train-vigor.json --batch_size 16  --name vigor_same

# python -u train.py --config models/config/ZOD/train.json --batch_size 16  --name zod_same

# Sinteract -p gpu -g gpu:1 -t 02:00:00 -m 50G -c 20