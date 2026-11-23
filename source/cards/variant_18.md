№18. EuroSAT + SmallCNN scratch
Суть: EuroSAT, SmallCNN scratch; фокус — окупается ли ImageNet-перенос на
не-ImageNet домене. Вопрос: окупается ли ImageNet-перенос на домене,
непохожем на ImageNet? : сравнить с №1 (CIFAR) — где перенос полезнее.

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1018. eurosat. EDA + contact
sheet. 2. Классика (ст. 2). Лист обработки. HOG + LinearSVC: --classical hog.
Scratch CNN vs классика на не-ImageNet домене. 3. Аугментации (ст. 3). basic.
Абляция none vs basic. 4. Модель (ст. 4). smallcnn scratch. Params/память. 5.
Обучение (ст. 5). SGD lr 0.1 cosine, 30 эпох. Кривые, resume. 6. Оценка (ст. 6).
Полная + robustness. 7. Grad-CAM (ст. 7). 5+5; объект vs фон. 8. Перенос (ст. 8).
Сравнение с №17 (scratch vs transfer на спутниках): таблица scratch vs probe
vs FT; бонус — сравнение с CIFAR (№1). 9. Отчёт (ст. 9). Latency, cost, ответ на
вопрос (окупается ли перенос), таблица классика/scratch/transfer.

Команды:

python train.py --dataset eurosat --model smallcnn --aug basic --optimizer sgd --lr 0.1 \
 --scheduler cosine --epochs 30 --seed 1018 --out_dir drive/out-var18
python evaluate.py --ckpt drive/out-var18/best.pt --dataset eurosat --model smallcnn \
 --split val --seed 1018 --classical hog --processing --gradcam --robustness --latency --out_dir
drive/eval-var18

Артефакты: таблица scratch vs transfer (с №17), сравнение с CIFAR, отчёт.
Ловушка: сравнивать scratch и transfer на разных сплитах EuroSAT —
используй один seed/сплит.

