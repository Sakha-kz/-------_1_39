№26. GTSRB (дорожные знаки) + ResNet-20
Суть: GTSRB (43 знака), ResNet-20; фокус — safety-critical ошибки. Вопрос: цена
ошибки: какие путаницы опасны и как их снизить (порог уверенности)? :
rejection option — отказ при низкой уверенности, кривая coverage/accuracy.

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1026. gtsrb (43 класса, 48×48).
EDA + contact sheet. 2. Классика (ст. 2). Лист обработки. HOG + LinearSVC +
цветовые маски (красный/синий): код блока 4 (HSV-маски + HOG).
Классический до-DL пайплайн знаков. 3. Аугментации (ст. 3). basic. Абляция
none vs basic. 4. Модель (ст. 4). resnet20. Params/память. 5. Обучение (ст. 5).
SGD lr 0.1 cosine, 30 эпох. Кривые, resume. 6. Оценка (ст. 6). Полная +

robustness; фокус — опасные пары (похожие знаки, ограничения скорости).
7. Grad-CAM (ст. 7). 5+5; объект vs фон. 8. Перенос (ст. 8). Дообучение на
искажениях (темнота/дождь/blur): fine-tune с corruption-аугментациями;
замер до/после на опасных парах. 9. Отчёт (ст. 9). Latency, cost, ответ на
вопрос (цена ошибки), кривая coverage/accuracy (бонус), таблица
классика/scratch/transfer.

Команды:

python train.py --dataset gtsrb --model resnet20 --aug basic --optimizer sgd --lr 0.1 \
 --scheduler cosine --epochs 30 --seed 1026 --out_dir drive/out-var26
python evaluate.py --ckpt drive/out-var26/best.pt --dataset gtsrb --model resnet20 \
 --split val --seed 1026 --processing --gradcam --robustness --latency --out_dir drive/eval-
var26

Артефакты: список опасных пар, кривая coverage/accuracy, цветовые маски
(код), отчёт. Ловушка: смотреть только на accuracy — для safety-critical важны
конкретные опасные пары, а не среднее.

