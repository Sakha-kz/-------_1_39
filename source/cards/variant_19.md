№19. UC Merced + ResNet-18
Суть: UC Merced (аэроснимки), ResNet-18; фокус — симметричность переноса
между аэродоменами. Вопрос: симметричен ли перенос между
аэродоменами? : объединённый датасет — мультидоменное обучение.

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1019. ucmerced (вручную,
ImageFolder root/ucmerced/train|test/<класс>). EDA + contact sheet. 2. Классика (ст.
2). Лист обработки. HOG + colorhist + LinearSVC: код блока 4, concat. Классика
как домен-независимая точка отсчёта. 3. Аугментации (ст. 3). basic. Абляция
none vs basic. 4. Модель (ст. 4). resnet18 + ImageNet. Params/память. 5. Обучение
(ст. 5). Transfer probe→FT 10+10 (блок 9), --norm imagenet --size 224. Кривые,
resume. 6. Оценка (ст. 6). Полная + robustness. 7. Grad-CAM (ст. 7). 5+5; объект
vs фон. 8. Перенос (ст. 8). domain shift → EuroSAT (обратное направление
№17): оцени на eurosat, дообучи; сравни симметричность переноса A→B vs
B→A. 9. Отчёт (ст. 9). Latency, cost, ответ на вопрос (симметрия), таблица
классика/scratch/transfer.

Команды:

python train.py --dataset ucmerced --model resnet18 --pretrained --aug basic --norm
imagenet \
 --size 224 --optimizer sgd --lr 1e-3 --scheduler cosine --epochs 10 --seed 1019 --out_dir
drive/out-var19-ft
python evaluate.py --ckpt drive/out-var19-ft/best.pt --dataset ucmerced --model resnet18 \
 --norm imagenet --size 224 --split val --seed 1019 --processing --gradcam --robustness --
latency --out_dir drive/eval-var19

Артефакты: таблица переноса A→B vs B→A (с №17), мультидомен (бонус),
отчёт. Ловушка: без ручной загрузки UC Merced датасет не откроется —
заложи время на скачивание и раскладку по папкам.

