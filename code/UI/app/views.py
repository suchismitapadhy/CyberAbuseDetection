from app import app, models
from .models import *
from flask import render_template, Flask, redirect, url_for, session, request, escape


@app.route("/")
@app.route("/index")
def index():
    '''
    Displays the home page
    '''
    return render_template("index.html")


@app.route("/check-for-abuse", methods=["POST"])
def check_for_abuse():
    '''
    Checks if entered text contains abusive context
    '''
    if request.method == "POST":
        abusive_text = request.form["abusive-text"]
        try:
            confidence = models.is_abuse(abusive_text)
        except:
            return render_template("index.html", error=True)
        return render_template("index.html", abusive=confidence)


@app.route("/analyze-twitter")
def display_twitter_page():
    '''
    Opens the page for the user to enter the twitter handle
    '''
    return render_template("twitterAnalysis.html")


@app.route("/check-twitter-handle", methods=["POST"])
def check_twitter_handle():
    '''
    Performs analysis on the input twitter handle
    '''
    if request.method == "POST":
        twitter_handle = request.form["twitter-handle"]
        try:
            abusive_tweets, recent_tweets = models.retrieve_abusive_tweets(twitter_handle)
        except:
            return render_template("twitterAnalysis.html", error=True)
        return render_template("twitterAnalysis.html", abusive_tweets=abusive_tweets, recent_tweets=len(recent_tweets))
