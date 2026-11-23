№12. SVHN + robustness
Суть: SVHN, SmallCNN scratch; фокус — устойчивость к искажениям
(corruptions). Вопрос: что эффективнее против искажений: аугментации при
обучении или дообучение? : кривая accuracy vs σ шума (5 точек).

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1012. svhn. EDA + contact
sheet. 2. Классика (ст. 2). Лист обработки. Canny-границы + HOG по границам
+ LinearSVC: код блока 4 (canny → hog на карте границ). Робастность
классических признаков к шуму/blur. 3. Аугментации (ст. 3). Базово none;
абляция none vs noise/blur-аугментации. Фокус — corruptions. 4. Модель (ст.
4). smallcnn. Params/память. 5. Обучение (ст. 5). SGD lr 0.1 cosine, 30 эпох.
Кривые, resume. 6. Оценка (ст. 6). Полная + расширенный robustness:
шум/blur/яркость + JPEG-артефакты; падение acc на каждом. 7. Grad-CAM (ст.
7). 5+5; объект vs фон/артефакт. 8. Перенос (ст. 8). Дообучение на
искажённых данных: fine-tune на train с corruption-аугментациями; сравни
«аугментации при обучении» vs «дообучение». 9. Отчёт (ст. 9). Latency, cost,
ответ на вопрос (что эффективнее), кривая acc vs σ (бонус), таблица
классика/scratch/robust-FT.

Команды:

python train.py --dataset svhn --model smallcnn --aug none --optimizer sgd --lr 0.1 \
 --scheduler cosine --epochs 30 --seed 1012 --out_dir drive/out-var12
python evaluate.py --ckpt drive/out-var12/best.pt --dataset svhn --model smallcnn \
 --split val --seed 1012 --processing --gradcam --robustness --latency --out_dir drive/eval-
var12

Артефакты: таблица corruptions (вкл. JPEG), кривая acc vs σ, сравнение
аугментации vs дообучение, отчёт. Ловушка: Canny-HOG через --classical не

собрать — это код блока 4; и не забудь JPEG-артефакты (их нет в базовом
robustness).

