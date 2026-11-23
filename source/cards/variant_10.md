№10. EMNIST letters + LeNet ★
Суть: EMNIST letters (26 классов, естественный дисбаланс), LeNet scratch; фокус
— дисбаланс и macro-F1. Вопрос: как дисбаланс классов искажает accuracy и
что лечит macro-F1? : focal loss третьим методом.

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1010. emnist (letters, 26
классов). EDA: гистограмма частотности букв (дисбаланс виден сразу). 2.
Классика (ст. 2). Лист обработки. HOG + LinearSVC (balanced): код блока 4,
balanced=True. Сравни accuracy vs macro-F1 у классики. 3. Аугментации (ст. 3).
rotate10 (буквы — без flip). Абляция none vs rotate10. 4. Модель (ст. 4). lenet.
Params/память. 5. Обучение (ст. 5). SGD lr 0.1 cosine, 20 эпох. Кривые, resume.
6. Оценка (ст. 6). Полная + robustness; главная метрика — macro-F1, per-class
P/R/F1, confusion matrix. 7. Grad-CAM (ст. 7). 5+5; объект vs фон. 8. Перенос (ст.
8). imbalance: weighted loss (nn.CrossEntropyLoss(weight=...)) vs oversampling
(WeightedRandomSampler); per-class F1 до/после. Бонус — focal loss. 9. Отчёт (ст.
9). Latency, cost, ответ на вопрос (accuracy vs macro-F1), таблица
классика/scratch/imbalance.

Команды:

python train.py --dataset emnist --model lenet --aug rotate10 --optimizer sgd --lr 0.1 \
 --scheduler cosine --epochs 20 --seed 1010 --out_dir drive/out-var10
python evaluate.py --ckpt drive/out-var10/best.pt --dataset emnist --model lenet \
 --split val --seed 1010 --processing --gradcam --robustness --latency --out_dir drive/eval-
var10

Артефакты: гистограмма частот, таблица accuracy vs macro-F1, per-class F1
до/после, отчёт. Ловушка: судить о качестве по accuracy при дисбалансе —
редкие буквы «тонут»; смотри macro-F1.

