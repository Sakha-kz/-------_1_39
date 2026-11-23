№30. Синтетика: фигуры + MLP ★ (CPU-friendly)
Суть: синтетические фигуры, MLP; фокус — пространственная
инвариантность CNN vs MLP. Вопрос: CNN vs MLP: откуда берётся
пространственная инвариантность? : визуализировать, где MLP ломается
(heatmap ошибок по позициям).

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1030. synthetic. EDA + contact
sheet. 2. Классика (ст. 2). Лист обработки. моменты Hu vs raw pixels + kNN:
код блока 4, clf='knn', два kind (hu, raw). Откуда инвариантность в классике? 3.
Аугментации (ст. 3). none vs basic (сдвиг/масштаб). Абляция обязательна. 4.
Модель (ст. 4). mlp. Params/память; выровняй с №29 (SmallCNN) для честного
сравнения. 5. Обучение (ст. 5). SGD lr 0.1 cosine, 20 эпох. Кривые, resume. 6.
Оценка (ст. 6). Полная + robustness; фокус — инвариантность
(сдвиг/масштаб/поворот). 7. Grad-CAM (ст. 7). Для MLP — saliency; 5+5 ошибок.
8. Перенос (ст. 8). Сравнение с №29 (CNN vs MLP): таблица инвариантности;
heatmap ошибок по позициям (бонус). 9. Отчёт (ст. 9). Latency (CPU), cost, ответ
на вопрос (инвариантность), таблица классика/MLP/CNN.

Команды:

python train.py --dataset synthetic --model mlp --aug basic --optimizer sgd --lr 0.1 \
 --scheduler cosine --epochs 20 --seed 1030 --out_dir drive/out-var30
python evaluate.py --ckpt drive/out-var30/best.pt --dataset synthetic --model mlp \
 --split val --seed 1030 --processing --robustness --latency --out_dir drive/eval-var30

Артефакты: heatmap ошибок по позициям, таблица Hu vs raw + kNN,
сравнение с №29, отчёт. Ловушка: MLP без аугментаций сдвига «ломается» на
новых позициях — это и есть предмет вопроса, покажи heatmap, а не
скрывай.

