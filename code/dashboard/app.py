from flask import Flask, render_template

app = Flask(__name__, template_folder='../templates')

#test airport data with image URLs
airport_data = [
    {
        "name": "Abuja Nnamdi Azikwe International Airport",
        "rating": "2",
        "image": "abuja.jpg"  #image
    },
    {
        "name": "Accra Kotoka International Airport",
        "rating": "3",
        "image": "accra.jpg"  #image
    },
    {
        "name": "Addis Ababa Bole International Airport",
        "rating": "2",
        "image": "addis_ababa.jpg"  #image
    }
]

@app.route('/')
def home():
    return render_template('index.html', data=airport_data)

@app.route('/airport/<int:index>')
def airport_details(index):
    #airport details
    if 0 <= index < len(airport_data):
        airport = airport_data[index]
        return render_template('airport_details.html', airport=airport)
    else:
        return "Airport not found.", 404

if __name__ == "__main__":
    app.run(debug=True)