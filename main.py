from flask_book import create_app, db
import os
from flask_migrate import Migrate
import torch
import gc
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()
gc.collect()

app = create_app()
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'users.db')
migrate = Migrate(app, db)

if __name__ == '__main__':
    if not os.path.exists(db_path):
        with app.app_context():
            db.create_all()
            print("Created database")
    app.run(debug=True) 