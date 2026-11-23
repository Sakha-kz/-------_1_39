№23. Food-101 + ResNet-18 + cutmix
Суть: Food-101 subset 10 (тот же, что у №22), ResNet-18 + cutmix; фокус — mixup
vs cutmix. Вопрос: mixup vs cutmix: что лучше на мелких текстурах и почему?
: визуализировать смешанные батчи обоих методов.

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1023. food101, тот же subset
10, что у №22. EDA + contact sheet. 2. Классика (ст. 2). Лист обработки. LBP +
LinearSVC: --classical lbp. Кто устойчивее к смешиванию текстур? 3.
Аугментации (ст. 3). basic + cutmix (--cutmix 1.0). Абляция none vs cutmix. 4.
Модель (ст. 4). resnet18 + ImageNet. Params/память. 5. Обучение (ст. 5). Transfer
probe→FT 10+10 (блок 9) с cutmix, --norm imagenet --size 224. Кривые, resume. 6.

Оценка (ст. 6). Полная + robustness; сравни с mixup (№22). 7. Grad-CAM (ст. 7).
5+5; объект vs фон. 8. Перенос (ст. 8). transfer + FT с cutmix; сравнение mixup
(№22): таблица cutmix vs mixup. 9. Отчёт (ст. 9). Latency, cost, ответ на вопрос
(cutmix vs mixup), визуализация батчей (бонус), таблица
классика/scratch/transfer.

Команды:

python train.py --dataset food101 --model resnet18 --pretrained --aug basic --cutmix 1.0 \
 --norm imagenet --size 224 --optimizer sgd --lr 1e-3 --scheduler cosine --epochs 10 --seed
1023 --out_dir drive/out-var23-ft
python evaluate.py --ckpt drive/out-var23-ft/best.pt --dataset food101 --model resnet18 \
 --norm imagenet --size 224 --split val --seed 1023 --classical lbp --processing --gradcam --
robustness --latency --out_dir drive/eval-var23

Артефакты: визуализация cutmix/mixup батчей, таблица сравнения, отчёт.
Ловушка: cutmix и mixup одновременно не включай (это отдельный режим);
для сравнения — два разных прогона.

