# מדריך אימון מודלים למסחר פיננסי

## מה קורה בתהליך האימון?

### 1. הכנת הנתונים
- טעינת נתוני מניות מ-`data/training_data.csv`
- יצירת רצפי זמן של 30 צעדים עבור כל מניה
- חישוב תשואות ותנודתיות
- חלוקה לאימון (80%) ובדיקה (20%)

### 2. מודלים שמתאמנים
הסקריפט מאמן 6 מודלים שונים בו-זמנית:

1. **LSTM קטן** - 4 שכבות, 128 נוירונים
2. **LSTM גדול** - 6 שכבות, 256 נוירונים  
3. **Transformer קטן** - 4 שכבות, 256 ממדים
4. **Transformer גדול** - 8 שכבות, 512 ממדים
5. **Ensemble Model** - שילוב של מספר מודלים
6. **WaveNet** - ארכיטקטורה מתקדמת לרצפים

### 3. תהליך האימון
- **אופטימיזר**: AdamW עם decay
- **Learning Rate**: מתחיל מ-1e-3 עם OneCycleLR scheduling
- **Loss Function**: פונקציה מותאמת לפיננסים (תשואה + כיוון + תנודתיות)
- **Early Stopping**: עוצר אם אין שיפור ב-15 epochs
- **Gradient Clipping**: מונע gradients גדולים מדי

## מה נשמר במהלך האימון?

### תיקיית `experiments/{experiment_name}/`

#### `checkpoints/` - נקודות שמירה
```
checkpoint_batch_100_advanced_lstm_small.pth
checkpoint_batch_200_advanced_lstm_small.pth
...
best_advanced_lstm_small.pth
best_financial_transformer_large.pth
```

**כל checkpoint מכיל:**
- משקלי המודל
- מצב האופטימיזר
- מצב ה-scheduler
- מספר האפוק והבאצ'
- היסטוריית הלוסים והדיוק

#### `plots/model_comparison.png`
גרף השוואה מקיף:
- ביצועי כל המודלים
- יעילות פרמטרים
- עקומות אימון
- דירוג מודלים

#### `model_comparison.json`
תוצאות מפורטות לכל מודל:
```json
{
  "advanced_lstm_small": {
    "best_val_loss": 0.034567,
    "final_val_acc": 0.5234,
    "param_count": 147456,
    "config": {...}
  }
}
```

## איך להריץ את האימון?

### אימון רגיל
```bash
python train_advanced.py
```

### אימון עם שם מתואם אישית
```bash
python train_advanced.py --experiment my_trading_models
```

### בדיקת מצב המודלים
```bash
python train_advanced.py --check
```

## מה קורה כשמפסיקים את האימון?

**המערכת שומרת אוטומטית:**
- Checkpoint כל 100 batches
- המודל הטוב ביותר בכל epoch
- כל ההיסטוריה של האימון

**כשמפעילים שוב:**
- מזהה אוטומטית checkpoints קיימים
- ממשיך מהנקודה האחרונה
- לא מאבד מידע או התקדמות

## איך לדעת אם המודל טוב?

### מדדי איכות
- **דיוק מעל 55%** - מעולה, מוכן לייצור
- **דיוק 52-55%** - טוב, שקול שיפורים  
- **דיוק מתחת ל-52%** - זקוק לשיפור

### מה לבדוק בפלט
```
🤖 FINANCIAL_TRANSFORMER_LARGE
   🎯 Validation Accuracy: 0.5687 (56.87%)
   📉 Best Val Loss: 0.034567
   📈 Quality: 🟢 Excellent
```

## פתרון בעיות נפוצות

### "No training data found"
- ודא שהקובץ `data/training_data.csv` קיים
- הרץ את סקריפט הכנת הנתונים קודם

### "CUDA out of memory"
- הקטן את batch_size בקוד
- סגור תוכנות אחרות
- השתמש במודלים קטנים יותר

### "Training stopped early"
- בדוק שיש מספיק נתונים
- שקול להקטין learning rate
- בדוק איכות הנתונים

## המלצות לאימון טוב

1. **הכן נתונים איכותיים** - יותר מניות, תקופות ארוכות יותר
2. **התאמן על GPU** - פי 10-20 מהר יותר מ-CPU
3. **עקב אחר התקדמות** - השתמש ב-`--check` לעדכונים
4. **בדוק overfitting** - השווה train vs validation accuracy
5. **נסה hyperparameters שונים** - ערוך את הקונפיגורציה במודלים

## קבצים חשובים

- `train_advanced.py` - הסקריפט הראשי
- `advanced_models.py` - הגדרות המודלים
- `data/training_data.csv` - נתוני האימון
- `experiments/` - תוצאות ומודלים שמורים

---

**טיפ:** הרץ תחילה `python train_advanced.py --check` כדי לראות אם יש מודלים קיימים לפני שמתחיל אימון חדש.