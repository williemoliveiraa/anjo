from flask import Flask
import os

app = Flask(__name__)

def led_init():
    cmd = 'python3.7 led_init.py'
    os.system(cmd)

def led_bed_entry():
    cmd = 'python3.7 led_bed_entry.py'
    os.system(cmd)

def led_hands():
    cmd = 'python3.7 led_hands.py'
    os.system(cmd)

def led_person_on_site():
    cmd = 'python3.7 led_person_on_site.py'
    os.system(cmd)

@app.route("/init")
def init():
    led_init()
    return "init"

@app.route("/bed_entry")
def bed_entry():
    led_bed_entry()
    return "bed_entry"

@app.route("/hands")
def hands():
    led_hands()
    return "hands"

@app.route("/person")
def hands():
    led_person_on_site()
    return "person"

app.run(port=8080)
