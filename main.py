import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

import pandas as pd

with open("intents.json") as  file:
    data = json.load(file)

# Given data
d = [
    {
        "title": "Cook-off by CodeChef",
        "about": "Cook-off is an amazing 2.5 hours long coding competition held by Codechef every month. This competition helps improve your analytical and problem-solving skills.",
        "siteUrl": "https://www.codechef.com/COOK134",
        "YouTube": "https://www.youtube.com/playlist?list=PLQXZIFwMtjoxrJvVaqoGlFYJRwUCHUq1t",
    },
    {
        "title": "Code Jam to I/O for Women",
        "about": "At Google, we're committed to building for everyone, and we know that a diversity of voices leads to better outcomes. We’re committed to increasing representation and building community in the online coding contest space and at Google I/O, our largest developer conference. Code Jam to I/O for Women is one way we bring women (students and professionals) from around the globe together, working to solve tough algorithmic challenges in a 2.5-hour, single-round coding competition. The top 150 on the scoreboard will receive a ticket and a stipend to participate in virtual Google I/O.",
        "siteUrl": "https://codingcompetitions.withgoogle.com/codejamio",
        "YouTube": "https://www.youtube.com/watch?v=Q_2TwBhqHPg",
    },
    {
        "title": "Code Gladiators",
        "about": "Code Gladiators is an annual coding competition by TechGig, that draws the best and the brightest coding talent from all parts of India. With multiple contests in emerging technologies and the coveted title of Code Gladiators up for grabs, the competition sees enthusiastic participation and has grown from strength to strength with each passing year. The last date to register in this contest is June 20, 2022.",
        "siteUrl": "https://www.techgig.com/codegladiators",
        "YouTube": "https://youtu.be/fzymgR7EdUs",
    },
    {
        "title": "FOSSASIA Codeheat",
        "about": "Codeheat is a coding contest for FOSSASIA projects on GitHub. The contest is separated into two months period after which winners of each period are announced. The jury chooses the winners from the top 10 contributors according to code quality and relevance of commits for the project each period. The jury also takes other contributions like submitted scrum reports and technical blog posts into account, but of course awesome code is the most important item on the list. Other participants have the chance to win T-shirts and Swag and get certificates of participation.",
        "siteUrl": "https://codeheat.org/",
        "YouTube": "https://www.youtube.com/watch?v=7jD6Iy-1EOs",
    },
    {
        "title": "Google Code Jam",
        "about": "Google Code Jam is conducted by Google from 2003. The competition consists of a set of algorithmic problems which must be solved in a fixed amount of time. The winner is awarded $15000 and there are smaller prizes for the runner ups.",
        "siteUrl": "https://codingcompetitions.withgoogle.com/codejam",
        "YouTube": "https://www.youtube.com/watch?v=cpguolx2oms",
    },
    {
        "title": "Google Kick Start",
        "about": "Online coding contest with international participants. Solve quality algorithmic questions designed by engineers at Google. Rounds take place region-wise. Scoring is based on penalty time and total points earned. Points earned = Total score | Penalty time = Time taken to pass maximum test cases. Top participants might even get an interview opportunity at Google. Certificates are given to all participants who submit at least 1 question. Consecutive participation after a year will also display rank on the certificate.",
        "siteUrl": "https://codingcompetitions.withgoogle.com/codejam",
        "YouTube": "https://www.youtube.com/watch?v=cpguolx2oms",
    },
    {
        "title": "Global Coding Challenge by Credit Suisse",
        "about": "The Global Coding Challenge is an online coding competition between participants across the globe. Around 3 weeks, users will be able to attempt solutions to nine coding problems. Participants can improve their code as many times as they like during the competition. After the completion of the competition, the Leaderboards will lock and the Global Coding Champion will be announced shortly. The competition has been entirely designed, built, and run by Credit Suisse TAs. Competition is split across 7 regions: UK, USA & Canada, Europe, India, Southeast Asia, Switzerland, and the rest of the world. 9 original questions, ranging from easy to hard, to be answered using any of 6 popular programming languages. There are prizes for the best individual coder globally, the top 3 coders of all 7 regions. Don't miss the chance to grab exciting prizes including MacBook Pro, iPhone, iPad Pro and much more! In the past competitions, students who have participated and done well have also joined the Credit Suisse team on a Summer Internship or as Technical Analysts.",
        "siteUrl": "https://www.credit-suisse.com/pwp/hr/en/codingchallenge/#/",
        "YouTube": "https://youtu.be/9Q4RDyqgN2g",
    },
    {
        "title": "Hackerrank Contests",
        "about": "Hackerrank Conducts various coding contests which are based on the core knowledge of Data Structures and Algorithms and also for any specific language. They have a wide range of exciting quality prizes like MacBook too.",
        "siteUrl": "https://www.hackerrank.com/contests",
        "YouTube": "https://youtu.be/2t7jYYBW4NI",
    },
    {
        "title": "Leetcode Contests",
        "about": "Leetcode hosts weekly and biweekly competitions mostly centered around data structures and algorithms. Each contest has a variety of prizes to be won.",
        "siteUrl": "https://leetcode.com/contest/",
        "YouTube": "https://youtu.be/elUB_Ga82tM",
    },
    {
        "title": "Newton School Grand Coding Contest",
        "about": "Newton School Grand Coding Contest is an annual coding competition by Newton School. It is one of India's Premier Coding Challenges with prizes up to 10 Lac Rupees. Top coders of India and across the globe compete in the foremost competitive coding contest of the country. It also gives access to internships and job opportunities directly through Newton School, with their hiring partner network of 800+ companies.",
        "siteUrl": "https://www.newtonschool.co/coding-contest",
        "YouTube": "https://youtu.be/W8KXpxYK900",
    },
    {
        "title":"ICPC",
        "about":" It is an annual competitive programming competition among universities from around the world. The contest involves solving algorithmic and computational problems within a fixed time frame.",
        "siteUrl":"https://icpc.global/",
        "YouTube":"https://www.youtube.com/@ICPCNews",
    }
]

# Create a DataFrame
df = pd.DataFrame(d)

h = [
    {
        'title': "HackerCup by Facebook",
        'about': "Hacker Cup is Facebook's annual open programming competition. Open to participants around the world, participants are invited to apply problem-solving and algorithmic coding skills to advance through each year’s online rounds, win prizes, and have a chance to make it to the global finals and win the grand prize.",
        'siteUrl': "https://www.facebook.com/codingcompetitions/hacker-cup",
        'YouTube': "https://www.youtube.com/watch?v=SA91yNdx_e0",
    },
    {
        'title': "HackCbs 5.0 by Hack2Skill",
        'about': "hackCBS 5.0, a legacy being carried forward by like-minded individuals aims to collaborate the intellects of programmers, designers, application developers, tech-geeks, and newbies in the world of programming for the intensive development of a hack. At hackCBS, we help you turn your ideas into reality by providing a comforting and welcoming environment. You’ll have all the freedom to create a product, learn new things, and have hilariously funny moments with your friends. Moreover, we’ll offer you a chance to network with working professionals and hacker community leaders. You will never learn faster than you will at a hackathon.",
        'siteUrl': "https://hack2skill.com/hack/hackcbs",
        'YouTube': "https://www.youtube.com/watch?v=tDcEn5Mu6nE",
    },
    {
        'title': "EY TECHNATHON",
        'about': "Welcome to the 3rd edition of the EY Techathon, your opportunity to build a better world in the Metaverse. Use technology and your imagination to solve any one of the three real-world challenges: Engineering, Entertainment, Health Participants will get a chance to use cutting-edge technology and interact with EY leadership and industry veterans. Winners will get a chance to win exciting cash prizes and internship opportunities at EY. College students from every discipline, across the country can participate in this technology challenge.",
        'siteUrl': "https://www.ey.com/en_in/techathon-3",
    },
    {
        'title': "Smart India Hackathon",
        'about': "Smart India Hackathon is a nationwide initiative to provide students with a platform to solve some of the pressing problems we face in our daily lives, and thus inculcate a culture of product innovation and a mindset of problem-solving. The first four editions SIH2017, SIH2018, SIH2019 and SIH2020 proved to be extremely successful in promoting innovation out-of-the-box thinking in young minds, especially engineering students from across India.",
        'siteUrl': "https://sih.gov.in//",
    },
    {
        'title': "Cricket Code Champions Hack",
        'about': "Cricket Code Champions Hack is a hackathon that merges the thrills of the ICC Cricket World Cup with cutting-edge technology and creative innovation. This hackathon invites tech enthusiasts, developers, and cricket aficionados to collaborate on groundbreaking solutions inspired by the world of cricket.",
        'siteUrl': "https://www.hackerearth.com/challenges/hackathon/cricket-code-champions-hack/",
    }
]

hackathon = pd.DataFrame(h)

intern = pd.read_csv('intern.csv')
mum = intern[intern['Location'].str.contains('Mumbai')]
dhl = intern[intern['Location'].str.contains('Delhi')]

r = [
    {
        'title' : "Open Directory",
        'url' : "https://aiplex.lol/"
    },
    {
        'title' : "Linktree",
        'url' : "https://linktr.ee/curiousdevelopers.in"
    },
    {
        'title' : "Data Structures and algorithms",
        'url' : "https://www.youtube.com/watch?v=5_5oE5lgrhw&list=PLu0W_9lII9ahIappRPN0MCAgtOu3lQjQi"
    },
    {
        'title' : "Online Web Tutorials",
        'url' : "https://www.w3schools.com/"
    }
]

rsc = pd.DataFrame(r)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
out = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
        
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    out.append(output_row)

training = numpy.array(training)
out = numpy.array(out)

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 12)
net = tflearn.fully_connected(net, 12)
net = tflearn.fully_connected(net, len(out[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, out, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

def bow(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i,w in enumerate(words):
            if w==se:
                bag[i] = 1

    return numpy.array(bag)

def chat():
    print("You can talk with the chatbot by typing below. type quit to stop")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bow(inp, words)])
        result_index = numpy.argmax(results)
        tag = labels[result_index]

        for t in data["intents"]:
            if t["tag"] == tag:
                responses = t["responses"]

        print(random.choice(responses))
        if(tag=="competitions"):
            print(df)
        if(tag=="Mumbai"):
            print(mum[['Title','Apply']])
        if(tag=="Delhi"):
            print(dhl)
        if(tag=="resources"):
            print(rsc)
        if(tag=="hackathon"):
            print(hackathon)
chat()