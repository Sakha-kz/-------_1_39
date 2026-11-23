№25. QuickDraw (10 категорий) + SmallCNN ★
Суть: QuickDraw subset 10 (скетчи), SmallCNN; фокус — разреженные штрихи
vs формы. Вопрос: разреженные скетчи: какие категории распознаются хуже
(штрихи vs формы)? : сравнить с фотографическими доменами — сложность
скетчей.

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1025. quickdraw (вручную,
ImageFolder root/quickdraw/train|test/<класс>), отбери 10 категорий. EDA + contact
sheet. 2. Классика (ст. 2). Лист обработки (skeletonize особенно показателен).
raw pixels + kNN; skeletonize + HOG: код блока 4, два подхода. Штрихи vs
формы. 3. Аугментации (ст. 3). rotate20 (±20°, без flip). Абляция none vs rotate20.
4. Модель (ст. 4). smallcnn. Params/память. 5. Обучение (ст. 5). SGD lr 0.1 cosine,
30 эпох. Кривые, resume. 6. Оценка (ст. 6). Полная + robustness; per-class —
какие категории хуже. 7. Grad-CAM (ст. 7). 5+5; на какие штрихи смотрит
модель. 8. Перенос (ст. 8). domain shift → EMNIST: оцени на emnist, дообучи;
кривые адаптации. 9. Отчёт (ст. 9). Latency, cost, ответ на вопрос (штрихи vs
формы), сравнение с фото-доменами (бонус), таблица
классика/scratch/transfer.

Команды:

python train.py --dataset quickdraw --model smallcnn --aug rotate20 --optimizer sgd --lr 0.1 \
 --scheduler cosine --epochs 30 --seed 1025 --out_dir drive/out-var25
python evaluate.py --ckpt drive/out-var25/best.pt --dataset quickdraw --model smallcnn \
 --split val --seed 1025 --processing --gradcam --robustness --latency --out_dir drive/eval-
var25

Артефакты: skeletonize-лист, таблица raw+kNN vs skeleton+HOG, per-class,
отчёт. Ловушка: skeletonize+HOG и kNN — код блока 4, не --classical; QuickDraw
нужно скачать и разложить вручную.

