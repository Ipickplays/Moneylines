@echo off
cd /d "C:\SportsPrediction\sports-predictions"
python update_predictions.py
netlify deploy --prod --dir=.
echo "Deploy complete at %date% %time%"