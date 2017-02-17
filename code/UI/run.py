import os
from app import app
from app.models import *
port = int(os.environ.get('PORT', 8081))
app.run(debug=True,host='0.0.0.0', port=port)
