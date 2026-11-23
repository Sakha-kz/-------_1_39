№36. CIFAR-10 + дистилляция ★
Суть: CIFAR-10, teacher ResNet-50 → student SmallCNN; фокус — knowledge
distillation. Вопрос: сколько качества teacher передаётся student при ×50
меньших параметрах? : sweep температуры T ∈ {2, 4, 8}.

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1036. cifar10. EDA + contact
sheet. 2. Классика (ст. 2). Лист обработки. HOG + LinearSVC: --classical hog.
Третья точка: teacher vs student vs классика. 3. Аугментации (ст. 3). basic.
Абляция none vs basic. 4. Модель (ст. 4). Teacher resnet50 (transfer, обучить),
student smallcnn. Params обоих (×50 разница). 5. Обучение (ст. 5). Teacher:
transfer probe→FT. Student: scratch 30 эпох и distilled (блок 11 kd_loss, T=4, α=0.7).
Кривые, resume. 6. Оценка (ст. 6). Полная + robustness; таблица teacher /
student scratch / student distilled. 7. Grad-CAM (ст. 7). 5+5 для student; объект vs
фон. 8. Перенос (ст. 8). Knowledge distillation: soft targets T=4, α=0.7; сравни
student scratch vs distilled; sweep T (бонус). 9. Отчёт (ст. 9). Latency student vs
teacher, cost, ответ на вопрос (передача качества), таблица
классика/teacher/student.

Команды:

python train.py --dataset cifar10 --model resnet50 --pretrained --aug basic --norm imagenet \
 --size 224 --optimizer sgd --lr 1e-3 --scheduler cosine --epochs 10 --seed 1036 --out_dir
drive/out-var36-teacher
python evaluate.py --ckpt drive/out-var36-teacher/best.pt --dataset cifar10 --model resnet50 \
 --norm imagenet --size 224 --split val --seed 1036 --classical hog --processing --gradcam --
robustness --latency --out_dir drive/eval-var36

Артефакты: код kd_loss, таблица teacher/student scratch/distilled, sweep T,
отчёт. Ловушка: KD-loss без множителя T² — градиенты soft-targets занижены;
используй kd_loss из блока 11 как есть.

