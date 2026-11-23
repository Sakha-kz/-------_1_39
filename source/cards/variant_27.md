№27. GTSRB + MobileNetV3
Суть: GTSRB, MobileNetV3; фокус — latency/FPS для встраиваемой системы.
Вопрос: accuracy vs скорость для real-time: какой компромисс приемлем? :
квантизация INT8 — потеря качества vs выигрыш скорости.

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1027. gtsrb. EDA + contact
sheet. 2. Классика (ст. 2). Лист обработки. HSV-сегментация цвета + HOG +
LinearSVC: код блока 4 (сегментация → HOG). Сегментация как признак. 3.
Аугментации (ст. 3). basic. Абляция none vs basic. 4. Модель (ст. 4). mobilenet_v3
+ ImageNet. Params/память; обоснуй встраиваемый выбор. 5. Обучение (ст. 5).
Transfer probe→FT 10+10 (блок 9), --norm imagenet --size 224. Кривые, resume. 6.
Оценка (ст. 6). Полная + robustness; обязательный latency/FPS (batch=1). 7.
Grad-CAM (ст. 7). 5+5; объект vs фон. 8. Перенос (ст. 8). Сравнение с №26
(ResNet-20): таблица accuracy/FPS/params; бонус — INT8-квантизация. 9. Отчёт
(ст. 9). Latency/FPS, cost, ответ на вопрос (компромисс real-time), таблица
классика/scratch/transfer.

Команды:

python train.py --dataset gtsrb --model mobilenet_v3 --pretrained --aug basic --norm
imagenet \
 --size 224 --optimizer sgd --lr 1e-3 --scheduler cosine --epochs 10 --seed 1027 --out_dir
drive/out-var27-ft
python evaluate.py --ckpt drive/out-var27-ft/best.pt --dataset gtsrb --model mobilenet_v3 \
 --norm imagenet --size 224 --split val --seed 1027 --processing --gradcam --robustness --
latency --out_dir drive/eval-var27

Артефакты: таблица accuracy/FPS vs №26, INT8 до/после (бонус), HSV-
сегментация (код), отчёт. Ловушка: INT8-квантизация без замера потери
accuracy — вывод о «выигрыше» неполный; всегда до/после.

