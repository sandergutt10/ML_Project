# AST Code Completion для Python

Проект по дисциплине «Прикладные методы искусственного интеллекта». Проект обучает трансформерную языковую модель на AST-подобных токенах и использует её для продолжения незавершённых фрагментов Python-кода.

## Цель проекта

Задача проекта — построить воспроизводимый пайплайн для автодополнения Python-кода:

- преобразование исходного Python-кода в AST-структуру датасета;
- обучение авторегрессионной модели на структурных токенах;
- восстановление корректного продолжения кода из сгенерированных AST-токенов;
- предоставление простого интерфейса для ручного инференса и демонстрации.

## Содержимое репозитория

```text
ML_Project/
|- checkpoints_code_lm/
|- docs/
|  |- report.pdf
|- frontend/
|  |- app.py
|- MLCode.py
|- parse_python.py
|- tests.py
|- requirements.txt
|- instruction.pdf
```

Основные файлы:

- `MLCode.py` — содержит конфигурацию, загрузку датасета, определение модели, цикл обучения, сохранение/загрузку чекпоинтов и CLI-команды.
- `parse_python.py` — содержит токенизацию, преобразование AST, вспомогательные функции для tolerant completion и утилиты восстановления кода.
- `tests.py` — запускает smoke-тесты на заранее подготовленных незавершённых фрагментах кода и сохраняет логи в `test_outputs/`.
- `frontend/app.py` — предоставляет Flask-веб-интерфейс для интерактивной демонстрации автодополнения.

## Установка

Требования:

- Python 3.11+
- `torch`
- `flask`
- `clearml` — для отслеживания экспериментов и подготовки итогового отчёта

Установка зависимостей:

```powershell
python -m pip install -r requirements.txt
```

Если вы планируете использовать отчёты ClearML, установите библиотеку отдельно:

```powershell
python -m pip install clearml
```

## Структура данных

По умолчанию обучение ожидает наличие следующих файлов в папке `data/`:

- `data/python100k_train.json`
- `data/python50k_eval.json`

Вы можете переопределить пути через CLI-аргументы.

## Загрузка датасета через KaggleHub

Для автоматической загрузки датасета можно использовать библиотеку `kagglehub`.

### 1. Установите необходимые зависимости

```powershell
python -m pip install kagglehub
```

### 2. Настройте доступ к Kaggle

1. Перейдите в настройки аккаунта Kaggle:
   https://www.kaggle.com/settings

2. В разделе **API** нажмите **Create New API Token**.

3. Файл `kaggle.json` будет скачан автоматически.

4. Поместите файл:
   - Windows:
     ```text
     C:\Users\<USERNAME>\.kaggle\kaggle.json
     ```
   - Linux / macOS:
     ```bash
     ~/.kaggle/kaggle.json
     ```

### 3. Загрузите датасет

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("glob4lh3ll/150k-python-dataset")

print("Path to dataset files:", path)
```

После загрузки датасет будет сохранён в локальный кэш KaggleHub. Скопируйте необходимые `.json` файлы в папку `data/`.

Пример структуры:

```text
data/
|- python100k_train.json
|- python50k_eval.json
```

## Запуск обучения

Минимальный локальный запуск:

```powershell
python MLCode.py --mode train
```

Запуск с пользовательскими путями к датасету и директорией чекпоинтов:

```powershell
python MLCode.py --mode train `
  --train-path data/python100k_train.json `
  --val-path data/python50k_eval.json `
  --checkpoint-dir checkpoints_code_lm
```

Полезные аргументы обучения:

- `--epochs`
- `--batch-size`
- `--accum-steps`
- `--seq-len`
- `--num-workers`
- `--resume-from`
- `--use-clearml`

## Запуск инференса

Генерация продолжения кода из сохранённого чекпоинта:

```powershell
python MLCode.py --mode infer `
  --checkpoint-path checkpoints_code_lm/best.pt `
  --sample-code "def add(a, b):`n    return "
```

## Smoke-тесты

Запуск smoke-тестов инференса:

```powershell
python tests.py
```

Примечания:

- `tests.py` ожидает наличие файла `checkpoints_code_lm/best.pt`.
- Если чекпоинт отсутствует, скрипт завершится с понятным сообщением вместо длинного traceback.
- Логи сохраняются в `test_outputs/`.

## Веб-интерфейс

Запуск frontend-интерфейса:

```powershell
python frontend/app.py
```

После запуска откройте `http://127.0.0.1:8000`.

Интерфейс использует тот же пайплайн инференса, что и CLI. В нём есть собственная веб-страница с удобным полем ввода кода, а также готовыми примерами из `tests.py`. Если `checkpoints_code_lm/best.pt` отсутствует, страница покажет понятное предупреждение о том, какой файл необходимо добавить.

## Работа в Google Colab

1. Загрузите папку проекта в Google Drive.
2. Поместите датасеты в папку `data/`.
3. Включите GPU в Colab.
4. Подключите Google Drive и перейдите в директорию проекта.
5. Установите зависимости.
6. Запустите обучение.
7. Продолжите обучение из `last.pt` или `best.pt`, если предыдущая сессия была прервана.
8. После обучения запустите smoke-тесты.

Пример команды обучения для Colab:

```bash
python MLCode.py \
  --mode train \
  --train-path data/python100k_train.json \
  --val-path data/python50k_eval.json \
  --checkpoint-dir checkpoints_code_lm \
  --epochs 12 \
  --batch-size 4 \
  --accum-steps 16 \
  --seq-len 1024 \
  --num-workers 0
```

## Текущие результаты

Текущее состояние проекта:

- реализован пайплайн обучения и инференса;
- реализована загрузка чекпоинтов и логика продолжения обучения в `MLCode.py`;
- реализованы smoke-тесты для качественной проверки автодополнения в `tests.py`;
- реализован интерактивный Flask-веб-интерфейс в `frontend/app.py`.

## Ограничения

- Репозиторий в текущем виде не содержит датасеты.
- Финальный `docs/report.pdf` необходимо экспортировать из ClearML вручную в соответствии с требованиями задания.
