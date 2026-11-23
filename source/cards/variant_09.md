№9. KMNIST + SmallCNN ★
Суть: KMNIST (кандзи), SmallCNN scratch; фокус — сложные штрихи и
визуальный разбор ошибок. Вопрос: сложные штрихи кандзи: где ошибки
(визуальный разбор 20 ошибок)? : сравнить с №5 (MNIST) — сложность
письменностей.

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1009. kmnist (28×28, 1 канал).
EDA + contact sheet. 2. Классика (ст. 2). Лист обработки. HOG + kNN: код блока
4, clf='knn'. Где ошибается классика на штрихах? 3. Аугментации (ст. 3).
rotate15 (±15°, без flip). Абляция none vs rotate15. 4. Модель (ст. 4). smallcnn.
Params/память. 5. Обучение (ст. 5). SGD lr 0.1 cosine, 30 эпох. Кривые, resume.
6. Оценка (ст. 6). Полная + robustness; визуальный разбор 20 ошибок. 7. Grad-
CAM (ст. 7). 5+5; на какие штрихи смотрит модель. 8. Перенос (ст. 8). domain
shift → EMNIST: оцени на emnist, дообучи; кривые forgetting/адаптации. 9.
Отчёт (ст. 9). Latency, cost, ответ на вопрос (разбор 20 ошибок), таблица
классика/scratch/transfer.

Команды:

python train.py --dataset kmnist --model smallcnn --aug rotate15 --optimizer sgd --lr 0.1 \
 --scheduler cosine --epochs 30 --seed 1009 --out_dir drive/out-var9

python evaluate.py --ckpt drive/out-var9/best.pt --dataset kmnist --model smallcnn \
 --split val --seed 1009 --processing --gradcam --robustness --latency --out_dir drive/eval-var9

Артефакты: разбор 20 ошибок с изображениями, таблица HOG+kNN,
сравнение с MNIST, отчёт. Ловушка: kNN через --classical не запустится —
нужен код блока 4.

