№7. Fashion-MNIST + SmallCNN ★
Суть: Fashion-MNIST, SmallCNN scratch; фокус — одежда vs цифры
(текстуры/формы). Вопрос: одежда vs цифры: что сложнее и почему
(текстуры/формы)? : per-class F1 — какие предметы одежды путаются.

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1007. fashion_mnist (28×28, 1
канал). EDA + contact sheet. 2. Классика (ст. 2). Лист обработки. HOG vs LBP +
LinearSVC: два прогона (--classical hog, --classical lbp). Форма или текстура
важнее? 3. Аугментации (ст. 3). basic (flip+crop допустим для одежды).
Абляция none vs basic. 4. Модель (ст. 4). smallcnn. Params/память. 5. Обучение
(ст. 5). SGD lr 0.1 cosine, 30 эпох. Кривые, resume. 6. Оценка (ст. 6). Полная +
robustness; per-class F1 (бонус) — какие предметы путаются. 7. Grad-CAM (ст.
7). 5+5; объект vs фон/текстура. 8. Перенос (ст. 8). domain shift → MNIST:
оцени на mnist, дообучи; кривые forgetting/адаптации. 9. Отчёт (ст. 9). Latency,
cost, ответ на вопрос (одежда vs цифры), таблица классика/scratch/transfer.

Команды:

python train.py --dataset fashion_mnist --model smallcnn --aug basic --optimizer sgd --lr 0.1 \
 --scheduler cosine --epochs 30 --seed 1007 --out_dir drive/out-var7
python evaluate.py --ckpt drive/out-var7/best.pt --dataset fashion_mnist --model smallcnn \
 --split val --seed 1007 --classical hog --processing --gradcam --robustness --latency --out_dir
drive/eval-var7

Артефакты: таблица HOG vs LBP, per-class F1, кривые domain shift, отчёт.
Ловушка: для LBP забыть нормировку гистограммы — она уже в шаблоне, не
дублируй и не ломай.

