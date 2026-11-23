№5. MNIST + LeNet-5 ★ (CPU-friendly)
Суть: MNIST, LeNet-5 scratch; фокус — потолок ~99% и разбор оставшегося 1%.
Вопрос: где потолок ~99% и что мешает оставшемуся 1% (разбор всех
ошибок)? : ensemble из 3 моделей — добить до 99.5%+.

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1005. mnist (28×28, 1 канал).
EDA + contact sheet. 2. Классика (ст. 2). Лист обработки. raw pixels + kNN vs
HOG + kNN: код блока 4, clf='knn', два kind (raw, hog). Где потолок классики? 3.
Аугментации (ст. 3). rotate10 (повороты ±10°, без flip — цифры). Абляция none
vs rotate10. 4. Модель (ст. 4). lenet (LeNet-5). Params/память; CPU-friendly. 5.
Обучение (ст. 5). SGD lr 0.1 cosine, 15 эпох, batch 128. Кривые, resume. 6.
Оценка (ст. 6). Полная + robustness; разбери ВСЕ ошибки на val (их немного).
7. Grad-CAM (ст. 7). 5+5; на какие штрихи смотрит модель. 8. Перенос (ст. 8).
domain shift → KMNIST: оцени MNIST-модель на kmnist, дообучи на KMNIST;
кривые forgetting/адаптации. 9. Отчёт (ст. 9). Latency (CPU), cost, ответ на
вопрос (потолок 99%), таблица классика/scratch/transfer.

Команды:

python train.py --dataset mnist --model lenet --aug rotate10 --optimizer sgd --lr 0.1 \
 --scheduler cosine --epochs 15 --seed 1005 --out_dir drive/out-var5
python evaluate.py --ckpt drive/out-var5/best.pt --dataset mnist --model lenet \
 --split val --seed 1005 --processing --gradcam --robustness --latency --out_dir drive/eval-var5

Артефакты: разбор всех ошибок, таблица raw+kNN vs HOG+kNN, кривые
domain shift, отчёт. Ловушка: применить flip к цифрам — часть ошибок
станет искусственной; используй rotate.

