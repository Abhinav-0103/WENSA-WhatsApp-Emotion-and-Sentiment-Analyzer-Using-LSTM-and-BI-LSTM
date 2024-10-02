# Import support labaraies
import emoji
import re
import matplotlib
import logging
import numpy as np
import pandas as pd
import altair as alt
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from typing import Any
from collections import Counter
from textblob import TextBlob
from urlextract import URLExtract
from wordcloud import WordCloud

extract = URLExtract()

def extract_emojis(s):
    """
        This function is used to calculate emojis in text and return in a list.
    """
    return [c for c in s if emoji.is_emoji(c)]

def fetch_stats(df):
    """
        This function is used to calculate statistics of the selected user.
    """
    # # Fetching for a particular User
    # if selected_user != 'All':
    #     df = df[df['user'] == selected_user]
    # else:
    #     df = df
    
    total_messages = df.shape[0]
    total_words = np.sum(df['total-words'])
    total_members = df['user'].nunique()
    total_media = np.sum(df['Media_Count'])
    total_urls = np.sum(df['total-urls'])

    total_emojis = np.sum(df['total-emojis'])
    avg_word_length = round(total_words/total_messages,2)

    df.dropna(subset=['date']) # NEW LINE ADDED

    if not df.empty:  # Check if the DataFrame is not empty
        last_date = df['date'].max().date().strftime("%d-%b-%y")
        first_date = df['date'].min().date().strftime("%d-%b-%y")
        most_active_day = df['day_name'].value_counts().idxmax()
        least_active_day = df['day_name'].value_counts().idxmin()
    else:
        # Handle the case when there are no valid dates in the DataFrame
        last_date = "N/A"
        first_date = "N/A"
        most_active_day = "N/A"
        least_active_day = "N/A"

    return {
            "media_message": total_media,
            "total_deleted_messages": len(df[df['message'] == "This message was deleted"]),
            "total_words": total_words,
            "total_messages": total_messages,
            'total_members': total_members,
            'link_shared': total_urls,
            "avg_word_length": avg_word_length,
            "total_emojis": total_emojis,
            "last_date": last_date,
            "first_date": first_date,
            "most_active_day": most_active_day,
            "least_active_day": least_active_day
        } 

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(columns={'index': 'name', 'user': 'percent'})
    return x,df

# Top 25 used Words during Conversation
def most_common_words(df):
    stop_words = []    
    try:
        f = open('./stop_hinglish.txt', 'r')
        stop_words = f.read()
    except FileNotFoundError:
        stop_words = ['.', '..', '...', '?', '-', '--', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'a', 'aadi', 'aaj', 'aap', 'aapne', 'aata', 'aati', 'aaya', 'aaye', 'ab', 'abbe', 'abbey', 'abe', 'abhi', 'able', 'about', 'above', 'accha', 'according', 'accordingly', 'acha', 'achcha', 'across', 'actually', 'after', 'afterwards', 'again', 'against', 'agar', 'ain', 'aint', "ain't", 'aisa', 'aise', 'aisi', 'alag', 'all', 'allow', 'allows', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'an', 'and', 'andar', 'another', 'any', 'anybody', 'anyhow', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'ap', 'apan', 'apart', 'apna', 'apnaa', 'apne', 'apni', 'appear', 'are', 'aren', 'arent', "aren't", 'around', 'arre', 'as', 'aside', 'ask', 'asking', 'at', 'aur', 'avum', 'aya', 'aye', 'baad', 'baar', 'bad', 'bahut', 'bana', 'banae', 'banai', 'banao', 'banaya', 'banaye', 'banayi', 'banda', 'bande', 'bandi', 'bane', 'bani', 'bas', 'bata', 'batao', 'bc', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'best', 'better', 'between', 'beyond', 'bhai', 'bheetar', 'bhi', 'bhitar', 'bht', 'bilkul', 'bohot', 'bol', 'bola', 'bole', 'boli', 'bolo', 'bolta', 'bolte', 'bolti', 'both', 'brief', 'bro', 'btw', 'but', 'by', 'came', 'can', 'cannot', 'cant', "can't", 'cause', 'causes', 'certain', 'certainly', 'chahiye', 'chaiye', 'chal', 'chalega', 'chhaiye', 'clearly', "c'mon", 'com', 'come', 'comes', 'could', 'couldn', 'couldnt', "couldn't", 'd', 'de', 'dede', 'dega', 'degi', 'dekh', 'dekha', 'dekhe', 'dekhi', 'dekho', 'denge', 'dhang', 'di', 'did', 'didn', 'didnt', "didn't", 'dijiye', 'diya', 'diyaa', 'diye', 'diyo', 'do', 'does', 'doesn', 'doesnt', "doesn't", 'doing', 'done', 'dono', 'dont', "don't", 'doosra', 'doosre', 'down', 'downwards', 'dude', 'dunga', 'dungi', 'during', 'dusra', 'dusre', 'dusri', 'dvaara', 'dvara', 'dwaara', 'dwara', 'each', 'edu', 'eg', 'eight', 'either', 'ek', 'else', 'elsewhere', 'enough', 'etc', 'even', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except', 'far', 'few', 'fifth', 'fir', 'first', 'five', 'followed', 'following', 'follows', 'for', 'forth', 'four', 'from', 'further', 'furthermore', 'gaya', 'gaye', 'gayi', 'get', 'gets', 'getting', 'ghar', 'given', 'gives', 'go', 'goes', 'going', 'gone', 'good', 'got', 'gotten', 'greetings', 'guys', 'haan', 'had', 'hadd', 'hadn', 'hadnt', "hadn't", 'hai', 'hain', 'hamara', 'hamare', 'hamari', 'hamne', 'han', 'happens', 'har', 'hardly', 'has', 'hasn', 'hasnt', "hasn't", 'have', 'haven', 'havent', "haven't", 'having', 'he', 'hello', 'help', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', "here's", 'hereupon', 'hers', 'herself', "he's", 'hi', 'him', 'himself', 'his', 'hither', 'hm', 'hmm', 'ho', 'hoga', 'hoge', 'hogi', 'hona', 'honaa', 'hone', 'honge', 'hongi', 'honi', 'hopefully', 'hota', 'hotaa', 'hote', 'hoti', 'how', 'howbeit', 'however', 'hoyenge', 'hoyengi', 'hu', 'hua', 'hue', 'huh', 'hui', 'hum', 'humein', 'humne', 'hun', 'huye', 'huyi', 'i', "i'd", 'idk', 'ie', 'if', "i'll", "i'm", 'imo', 'in', 'inasmuch', 'inc', 'inhe', 'inhi', 'inho', 'inka', 'inkaa', 'inke', 'inki', 'inn', 'inner', 'inse', 'insofar', 'into', 'inward', 'is', 'ise', 'isi', 'iska', 'iskaa', 'iske', 'iski', 'isme', 'isn', 'isne', 'isnt', "isn't", 'iss', 'isse', 'issi', 'isski', 'it', "it'd", "it'll", 'itna', 'itne', 'itni', 'itno', 'its', "it's", 'itself', 'ityaadi', 'ityadi', "i've", 'ja', 'jaa', 'jab', 'jabh', 'jaha', 'jahaan', 'jahan', 'jaisa', 'jaise', 'jaisi', 'jata', 'jayega', 'jidhar', 'jin', 'jinhe', 'jinhi', 'jinho', 'jinhone', 'jinka', 'jinke', 'jinki', 'jinn', 'jis', 'jise', 'jiska', 'jiske', 'jiski', 'jisme', 'jiss', 'jisse', 'jitna', 'jitne', 'jitni', 'jo', 'just', 'jyaada', 'jyada', 'k', 'ka', 'kaafi', 'kab', 'kabhi', 'kafi', 'kaha', 'kahaa', 'kahaan', 'kahan', 'kahi', 'kahin', 'kahte', 'kaisa', 'kaise', 'kaisi', 'kal', 'kam', 'kar', 'kara', 'kare', 'karega', 'karegi', 'karen', 'karenge', 'kari', 'karke', 'karna', 'karne', 'karni', 'karo', 'karta', 'karte', 'karti', 'karu', 'karun', 'karunga', 'karungi', 'kaun', 'kaunsa', 'kayi', 'kch', 'ke', 'keep', 'keeps', 'keh', 'kehte', 'kept', 'khud', 'ki', 'kin', 'kine', 'kinhe', 'kinho', 'kinka', 'kinke', 'kinki', 'kinko', 'kinn', 'kino', 'kis', 'kise', 'kisi', 'kiska', 'kiske', 'kiski', 'kisko', 'kisliye', 'kisne', 'kitna', 'kitne', 'kitni', 'kitno', 'kiya', 'kiye', 'know', 'known', 'knows', 'ko', 'koi', 'kon', 'konsa', 'koyi', 'krna', 'krne', 'kuch', 'kuchch', 'kuchh', 'kul', 'kull', 'kya', 'kyaa', 'kyu', 'kyuki', 'kyun', 'kyunki', 'lagta', 'lagte', 'lagti', 'last', 'lately', 'later', 'le', 'least', 'lekar', 'lekin', 'less', 'lest', 'let', "let's", 'li', 'like', 'liked', 'likely', 'little', 'liya', 'liye', 'll', 'lo', 'log', 'logon', 'lol', 'look', 'looking', 'looks', 'ltd', 'lunga', 'm', 'maan', 'maana', 'maane', 'maani', 'maano', 'magar', 'mai', 'main', 'maine', 'mainly', 'mana', 'mane', 'mani', 'mano', 'many', 'mat', 'may', 'maybe', 'me', 'mean', 'meanwhile', 'mein', 'mera', 'mere', 'merely', 'meri', 'might', 'mightn', 'mightnt', "mightn't", 'mil', 'mjhe', 'more', 'moreover', 'most', 'mostly', 'much', 'mujhe', 'must', 'mustn', 'mustnt', "mustn't", 'my', 'myself', 'na', 'naa', 'naah', 'nahi', 'nahin', 'nai', 'name', 'namely', 'nd', 'ne', 'near', 'nearly', 'necessary', 'neeche', 'need', 'needn', 'neednt', "needn't", 'needs', 'neither', 'never', 'nevertheless', 'new', 'next', 'nhi', 'nine', 'no', 'nobody', 'non', 'none', 'noone', 'nope', 'nor', 'normally', 'not', 'nothing', 'novel', 'now', 'nowhere', 'o', 'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'on', 'once', 'one', 'ones', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'own', 'par', 'pata', 'pe', 'pehla', 'pehle', 'pehli', 'people', 'per', 'perhaps', 'phla', 'phle', 'phli', 'placed', 'please', 'plus', 'poora', 'poori', 'provides', 'pura', 'puri', 'q', 'que', 'quite', 'raha', 'rahaa', 'rahe', 'rahi', 'rakh', 'rakha', 'rakhe', 'rakhen', 'rakhi', 'rakho', 'rather', 're', 'really', 'reasonably', 'regarding', 'regardless', 'regards', 'rehte', 'rha', 'rhaa', 'rhe', 'rhi', 'ri', 'right', 's', 'sa', 'saara', 'saare', 'saath', 'sab', 'sabhi', 'sabse', 'sahi', 'said', 'sakta', 'saktaa', 'sakte', 'sakti', 'same', 'sang', 'sara', 'sath', 'saw', 'say', 'saying', 'says', 'se', 'second', 'secondly', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves', 'sensible', 'sent', 'serious', 'seriously', 'seven', 'several', 'shall', 'shan', 'shant', "shan't", 'she', "she's", 'should', 'shouldn', 'shouldnt', "shouldn't", "should've", 'si', 'sir', 'sir.', 'since', 'six', 'so', 'soch', 'some', 'somebody', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'still', 'sub', 'such', 'sup', 'sure', 't', 'tab', 'tabh', 'tak', 'take', 'taken', 'tarah', 'teen', 'teeno', 'teesra', 'teesre', 'teesri', 'tell', 'tends', 'tera', 'tere', 'teri', 'th', 'tha', 'than', 'thank', 'thanks', 'thanx', 'that', "that'll", 'thats', "that's", 'the', 'theek', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'theres', "there's", 'thereupon', 'these', 'they', "they'd", "they'll", "they're", "they've", 'thi', 'thik', 'thing', 'think', 'thinking', 'third', 'this', 'tho', 'thoda', 'thodi', 'thorough', 'thoroughly', 'those', 'though', 'thought', 'three', 'through', 'throughout', 'thru', 'thus', 'tjhe', 'to', 'together', 'toh', 'too', 'took', 'toward', 'towards', 'tried', 'tries', 'true', 'truly', 'try', 'trying', 'tu', 'tujhe', 'tum', 'tumhara', 'tumhare', 'tumhari', 'tune', 'twice', 'two', 'um', 'umm', 'un', 'under', 'unhe', 'unhi', 'unho', 'unhone', 'unka', 'unkaa', 'unke', 'unki', 'unko', 'unless', 'unlikely', 'unn', 'unse', 'until', 'unto', 'up', 'upar', 'upon', 'us', 'use', 'used', 'useful', 'uses', 'usi', 'using', 'uska', 'uske', 'usne', 'uss', 'usse', 'ussi', 'usually', 'vaala', 'vaale', 'vaali', 'vahaan', 'vahan', 'vahi', 'vahin', 'vaisa', 'vaise', 'vaisi', 'vala', 'vale', 'vali', 'various', 've', 'very', 'via', 'viz', 'vo', 'waala', 'waale', 'waali', 'wagaira', 'wagairah', 'wagerah', 'waha', 'wahaan', 'wahan', 'wahi', 'wahin', 'waisa', 'waise', 'waisi', 'wala', 'wale', 'wali', 'want', 'wants', 'was', 'wasn', 'wasnt', "wasn't", 'way', 'we', "we'd", 'well', "we'll", 'went', 'were', "we're", 'weren', 'werent', "weren't", "we've", 'what', 'whatever', "what's", 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', "where's", 'whereupon', 'wherever', 'whether', 'which', 'while', 'who', 'whoever', 'whole', 'whom', "who's", 'whose', 'why', 'will', 'willing', 'with', 'within', 'without', 'wo', 'woh', 'wohi', 'won', 'wont', "won't", 'would', 'wouldn', 'wouldnt', "wouldn't", 'y', 'ya', 'yadi', 'yah', 'yaha', 'yahaan', 'yahan', 'yahi', 'yahin', 'ye', 'yeah', 'yeh', 'yehi', 'yes', 'yet', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've", 'yup']

    temp = df[df['name'] != 'group_notification']
    temp = temp[temp['message'] != '<media omitted>']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(25))
    return most_common_df

def emoji_extract(df):
    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if emoji.is_emoji(c)])

    # print(f"These are the emojis :- \n {emojis}\n\n\n\n")

    top10emojis = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis)))).sort_values(by = 1,ascending= False).head(20)
    top10emojis.rename(columns={0: 'emoji', 1: 'emoji_count'}, inplace=True)
    # print(f"This the top10emojis dataframe :- \n{top10emojis}\n\n\n\n\n")
    # print("This is the shape :- ", )
    top10emojis['emoji_description'] = [''] * top10emojis.shape[0]
    i = 0
    for item in top10emojis.emoji:
        description = emoji.demojize(item)[1:-1]
        top10emojis.emoji_description[i] = description
        i += 1
    d1 = top10emojis.head(10)
    d2 = top10emojis.tail(10)
    return d1,d2

# Monthly Timeline
def monthly_timeline(df):
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

# Which day has most msg
def week_activity_map(df2):
    return df2['day_name'].value_counts()

# Which month has highest MSG
def month_activity_map(df2):
    return df2['month'].value_counts()

# HeatMap of Weekday and Hour
def activity_heatmap_week(df):
    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    return user_heatmap

# HeatMap of Month and Day
def activity_heatmap_month(df3):
    months = ['Janurary', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    matplotlib.rcParams['font.size'] = 13
    matplotlib.rcParams['figure.figsize'] = (18, 8)

    df3['message_count'] = [1] * df3.shape[0]
    grouped_by_month_and_day = df3.groupby(['Month', 'Day']).sum().reset_index()[['Month', 'Day', 'message_count']]
    pt = grouped_by_month_and_day.pivot_table(index = 'Month', columns = 'Day', values = 'message_count').reindex(index = months, columns = days).fillna(0)
    return pt

def message_cluster(data_frame: pd.DataFrame):
    """
    Display Message Cluster base on the message statistics

    Attributes
    ----------
    Dataframe (pandas DF)

    Retrurns
    --------
    Matplotlib Figure
    """
    new_df = pd.DataFrame(data_frame[['message']].groupby(
        by=data_frame['name']).count())
    new_df['media_count'] = data_frame[['media']].groupby(
        by=data_frame['name']).sum()
    new_df['emoji_count'] = data_frame[['emojis']].groupby(
        by=data_frame['name']).sum()
    new_df['urlcount_count'] = data_frame[['urlcount']].groupby(
        by=data_frame['name']).sum()
    new_df['letter_count'] = data_frame[['letter_count']].groupby(
        by=data_frame['name']).sum()
    new_df['words_count'] = data_frame[['word_count']].groupby(
        by=data_frame['name']).sum()

    new_df.reset_index(level=0, inplace=True)
    fig = px.scatter(
        new_df, x="message", y="words_count",
        size="letter_count", color="name",
        hover_name="name", log_x=True, size_max=60)

    fig.update_layout(
        width=1200,
        height=550)
    return fig

def pie_display_emojis(data_frame: pd.DataFrame):
    
    """
    Pie chart formation for Emoji's Distrubution

    Attributes
    ----------
    Dataframe (pandas DF)

    Retrurns
    --------
    Plotly Figure (pyDash)
    """
    logging.info("WhatsApp/pie_display_emojis()")
    total_emojis_list = list(set([a for b in data_frame.emoji for a in b]))
    total_emojis_list = (a for b in data_frame.emoji for a in b)
    emoji_dict = dict(Counter(total_emojis_list))
    emoji_dict = sorted(emoji_dict.items(), key=lambda x: x[1], reverse=True)
    # for i in emoji_dict:
    #     print(i)
    emoji_df = pd.DataFrame(emoji_dict, columns=['emoji', 'count'])
    fig = px.pie(emoji_df, values='count', names='emoji')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        width=1100,
        height=550)
    
    return fig

def time_series_plot(df: pd.DataFrame):
    df2 = pd.DataFrame()
    df2['date'] = df['only_date'].values
    df2['message_count'] = [1] * df2.shape[0]      
    df2 = df2.groupby('date').sum().reset_index()
    fig = px.line(x=df2['date'], y=df2['message_count'], width= 1200)
    fig.update_layout(
        title="",
        xaxis_title='Conversation frequency during whole year',
        yaxis_title='Number of Messages'
        )
    fig.update_xaxes(nticks=70)
    return fig

def plot_data(data_string):
    """
    Common Bar chat Function for plotting data

    Attributes
    ----------
    Dataframe (pandas DF)

    Retrurns
    --------
    Matplotlib Figure
    """
    logging.info("WhatsApp/plot_data()")
    fig, ax_value = plt.subplots()
    # Save the chart so we can loop through the bars below.
    bars = ax_value.bar(
        x=np.arange(data_string.get('x_value')),
        height=data_string.get('y_value'),
        tick_label=data_string.get('tick_label'),
        color = data_string.get('bar_color')
    )

    # #all bar color random
    # for bar in bars:
    #     bar.set_color(np.random.rand(3,))

    # Axis formatting.
    ax_value.spines['top'].set_visible(False)
    ax_value.spines['right'].set_visible(False)
    ax_value.spines['left'].set_visible(False)
    ax_value.spines['bottom'].set_color('#686868')
    ax_value.tick_params(bottom=False, left=False)
    ax_value.tick_params(axis='x', labelrotation=40)
    ax_value.set_axisbelow(True)
    ax_value.yaxis.grid(True, color='#EEEEEE')
    ax_value.xaxis.grid(False)
    for bar_value in bars:
        ax_value.text(
            bar_value.get_x() + bar_value.get_width() / 2,
            bar_value.get_height()+2,
            round(bar_value.get_height(), 1),
            horizontalalignment='center',
            color=data_string.get('text_color')
        )
    ax_value.set_xlabel(
        data_string.get('x_label'), labelpad=15, color='#333333')
    ax_value.set_ylabel( 
        data_string.get('y_label'), labelpad=15, color='#333333')
    ax_value.set_title(
        data_string.get('title'), pad=15, color='#333333')
    return fig

def max_words_used(data_frame: pd.DataFrame):
    """
    Maximum words used in sentence in group chat

    Attributes
    ----------
    Dataframe (pandas DF)

    Retrurns
    --------
    Matplotlib Figure
    """
    logging.info("WhatsApp/max_words_used()")
    data_frame['word_count'] = data_frame['Message'].apply(lambda x: len(x.split()))
    max_words = data_frame[['User', 'word_count']].groupby('User').sum()
    m_w = max_words.sort_values('word_count', ascending=False).head(10)
    return plot_data({
            'x_value': m_w.size,
            'y_value': m_w.word_count,
            'tick_label': m_w.index,
            'x_label': 'Name of Group Member',
            'y_label': 'Number of Words in Group Chat',
            'title': 'Analysis of members who has used more words in his/her messages',
            'bar_color': 'pink',
            'text_color': 'black'
        })

def alt_air_plot(data_frame: pd.DataFrame):
    """
    Altair plot for group chat analysis

    Attributes
    ----------
    Dataframe (pandas DF)

    Retrurns
    --------
    Altair plot
    """
    logging.info("WhatsApp/alt_air_plot()")
    # Altlair of month wise messages
    alt_air_month = alt.Chart(data_frame).mark_bar().encode(
        x='Month',
        y='count:Q',
        color='User'
    ).properties(
        title='Month wise messages count',
        width=1200,
        height=550
    )
    # Altair of day wise messages

def most_active_member(df3,num):
    """
    Most active memeber as per number of messages in group

    Attributes
    ----------
    Dataframe (pandas DF)

    Retrurns
    --------
    Matplotlib Figure
    """
    logging.info("WhatsApp/most_active_member()")
    matplotlib.rcParams['font.size'] = 13
    matplotlib.rcParams['figure.figsize'] = (18, 10)
    mostly_active = df3['User'].value_counts()
    tot_user = df3['User'].nunique()
    if(tot_user >9):
        tot_user = num
    m_a = mostly_active.head(tot_user)
    return plot_data({
            'x_value': m_a.size,
            'y_value': m_a,
            'tick_label': m_a.index,
            'x_label': 'Name of Group Member',
            'y_label': 'Number of Group Messages',
            'title': 'Mostly Active member in Group (based on messages)',
            'bar_color': 'purple',
            'text_color': 'black'
        })

def top_10_days(df):
    """
    Top 10 days in which group chat was active

    Attributes
    ----------
    Dataframe (pandas DF)
    
    Retrurns
    --------
    Matplotlib Figure
    """
    df['message_count'] = [1] * df.shape[0]    
    logging.info("WhatsApp/top_10_days()")

    top_10_days = df['Date'].value_counts()
    top_10_days = top_10_days.head(10)
    return plot_data({
            'x_value': top_10_days.size,
            'y_value': top_10_days,
            'tick_label': top_10_days.index,
            'x_label': 'Date',
            'y_label': 'Number of Group Messages',
            'title': 'Top 10 days in which group chat was active',
            'bar_color': '#00BFFF',
            'text_color': 'black'
        })

def most_busy_hour(df):
    active_hour = df['period'].value_counts()
    matplotlib.rcParams['font.size'] = 14
    matplotlib.rcParams['figure.figsize'] = (25, 10)
    fig,ax = plt.subplots()
    bars = ax.bar(active_hour.index,active_hour.values,color='green')
    
    #setting all bars color randomly
    for bar in bars:
        bar.set_color(np.random.rand(3,))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#686868')
    ax.tick_params(bottom=False, left=False)
    ax.tick_params(axis='x', labelrotation=30)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)

    for bar_value in bars:
        ax.text(
            bar_value.get_x() + bar_value.get_width() / 2,
            bar_value.get_height()+2,
            round(bar_value.get_height(), 1),
            horizontalalignment='center',
            color='black'
        )

    ax.set_xlabel('Busy Hour (24 hour clock format)',labelpad=15, color='#333333')
    ax.set_ylabel('No of Messages',labelpad=15, color='#333333')
    return fig,ax

def most_busy_day(data_frame: pd.DataFrame):
    active_day = data_frame['day_name'].value_counts()
    a_d = active_day.head(7)
    return plot_data({
            'x_value': a_d.size,
            'y_value': a_d,
            'tick_label': a_d.index,
            'x_label': 'Name of days in a week',
            'y_label': 'Number of Messages',
            'title': 'Most active day of Week in the Group',
            'bar_color':'purple',
            'text_color':'black',
        })

def most_busy_month(df):
    active_month = df['month'].value_counts()
    a_m = active_month.head(12)
    return plot_data({
        'x_value': a_m.size,
        'y_value': a_m,
        'tick_label': a_m.index,
        'x_label': 'Name of months in year',
        'y_label': 'Number of Messages',
        'title': 'Most active months of year in the Group',
        'bar_color':'orange',
        'text_color':'blue'
        })
    
def top_media_contributor(data_frame: pd.DataFrame):
    """
    Top 10 members who shared media's in group

    Attributes
    ----------
    Dataframe (pandas DF)

    Retrurns
    --------
    Matplotlib Figure
    """
    logging.info("WhatsApp/top_media_contributor()")

    data_frame['media'] = data_frame['Message'].apply(lambda x: re.findall("omitted", x)).str.len()
    max_media = data_frame[['User', 'media']].groupby('User').sum()
    m_m = max_media.sort_values('media', ascending=False).head(15)
    data = {}
    for i in m_m.index:
        data[i] = m_m.loc[i, 'media']
    user = list(data.keys())
    mediaCount = list(data.values())
    reverse_user = user[::-1]
    reverse_mediaCount = mediaCount[::-1]

    fig = plt.figure(figsize = (22, 16))

    # Color each bar as random color
    colors = [np.random.rand(3,) for i in range(len(user))]
    plt.barh(reverse_user, reverse_mediaCount, color=colors)
    plt.xlabel("Number of Media's shared", fontsize=15)
    plt.ylabel("Name of Group Member", fontsize=15)   

    return fig

def getWebsite(url):
    ans = re.findall(r'(?P<url>http[s]?://[^/]+)', url)[0]
    ans = ans.replace('http://','')
    ans = ans.replace('https://','')
    ans = ans.replace('www.','')
    return ans

def fetch_url(msg_list):
    urlList = msg_list.apply(lambda x: re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', x))
    urlList = [x for x in urlList if x]
    return urlList

def fetch_url_list(msg_list):
    urlList = fetch_url(msg_list)
    websiteList = [getWebsite(str(x)) for x in urlList]
    websiteList = [x if x != 'www.youtube.com' and x != 'youtu.be' and x != 'youtube.com' else 'youtube.com' for x in websiteList]
    websiteCount = Counter(websiteList)
    websiteCount = websiteCount.most_common(20)
    df5 = pd.DataFrame(websiteCount, columns=['Website', 'Count'])
    df6 = df5.head(10)
    df7 = df5.tail(10)
    return df5,df6,df7

def url_bar_graph(df5):
    fig = plt.figure(figsize = (12, 9))
    colors = [np.random.rand(3,) for i in range(len(df5))]
    plt.barh(df5['Website'], df5['Count'], color=colors)
    plt.xlabel('Count of Website', fontsize=15)
    plt.ylabel('Name Of Website', fontsize=15)
    return fig

def url_line_graph(df6):
    fig = go.Figure(data=go.Scatter(x=df6['Website'], y=df6['Count'], mode='lines+markers+text', text=df6['Count'], textposition='top center', textfont=dict(color='black', size=14), showlegend=False))
    fig.update_traces(marker_size = 11, marker_line_color='rgb(0,0,0)', marker_line_width=2.5, opacity=0.6)
    fig.update_layout(width=1200, height=600)
    return fig

def url_scatter_bubble(df5):
    logging.info("WhatsApp/url_scatter_bubble()")
    fig = px.scatter(df5, x='Website', y='Count', size='Count', color='Count', hover_data=['Website', 'Count'], template="none")
    for i in range(len(fig.data)):
        fig.data[i].marker.color = fig.data[i].y
        fig.data[i].marker.size  = fig.data[i].y*7
    fig.update_layout(width=1200, height=600)
    return fig

def day_wise_pie(df3):
    """
    Day wise pie chart of group chat activity

    Attributes
    ----------
    Dataframe (pandas DF)
    
    Retrurns
    --------
    Matplotlib Figure
    """
    logging.info("WhatsApp/day_wise_pie()")
    tmp = df3['Day'].value_counts()
    day_name_count = dict(tmp)
    days = list(day_name_count.keys())
    counts = list(day_name_count.values())
    fig = go.Figure(
    go.Pie(
        labels = days,
        values = counts,
        hoverinfo = 'label+percent+value',
        # hoverinfo = "label+percent",
        textinfo = "value",
        textfont = dict(
            family = "Arial",
            size = 12,
            color = "black"
        )
        ))
    return fig

def day_wise_count(dataFrame):
    """
    This function generate a line polar plot.

    Parameters
    ----------
    data : DataFrame
        DESCRIPTION.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    """
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    day_df = pd.DataFrame(dataFrame['message'])
    day_df['day_of_date'] = dataFrame['date'].dt.dayofweek
    day_df['day_of_date'] = day_df["day_of_date"].apply(lambda d: days[d])
    day_df["messagecount"] = 1
    
    day = day_df.groupby("day_of_date").sum()
    day.reset_index(inplace=True)
    
    fig = px.line_polar(day, r='messagecount', theta='day_of_date', line_close=True)
    fig.update_traces(fill='toself')
    fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
        )),
    showlegend=False
    )
    return fig

def who_shared_links(data_frame: pd.DataFrame):
    """
    Top 10 members Who shared maximum links in Group

    Attributes
    ----------
    Dataframe (pandas DF)

    Retrurns
    --------
    Matplotlib Figure
    """
    logging.info("WhatsApp/who_shared_links()")
    data_frame['url_count'] = data_frame['Message'].str.count('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    max_words = data_frame[['User', 'url_count']].groupby('User').sum()
    m_w = max_words.sort_values('url_count', ascending=False).head(10)
    return plot_data({
            'x_value': m_w.size,
            'y_value': m_w.url_count,
            'tick_label': m_w.index,
            'x_label': 'Name of Group Member',
            'y_label': 'Number of Links Shared in Group',
            'title': 'Analysis of members who has shared max no. of links in Group',
            'bar_color':'orange',
            'text_color':'black'
        })

def time_when_group_active(data_frame: pd.DataFrame):
    """
    Most Messages Analsyis w.r.t to Time

    Attributes
    ----------
    Dataframe (pandas DF)

    Retrurns
    --------
    Matplotlib Figure
    """
    logging.info("WhatsApp/time_when_group_active()")
    # Time whenever the group was highly active
    active_time = data_frame.datetime.dt.time.value_counts().head(10)
    return plot_data({
            'x_value': active_time.size,
            'y_value': active_time.values,
            'tick_label': active_time.index,
            'x_label': 'Time',
            'y_label': 'Number of Messages',
            'title': 'Analysis of time when group was highly active'
        })

def cloud_data(df: pd.DataFrame) -> pd.DataFrame:
    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U0001F1F2-\U0001F1F4"  # Macau flag
            u"\U0001F1E6-\U0001F1FF"  # flags
            u"\U0001F600-\U0001F64F"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U0001F1F2"
            u"\U0001F1F4"
            u"\U0001F620"
            u"\u200d"
            u"\u2640-\u2642"
            "]+", flags=re.UNICODE)
    df['message'] = df.loc[:, 'message'].apply(
        lambda s: s.lower())\
        .apply(lambda s: emoji_pattern.sub(r'', s))\
        .str.replace('\n|\t', '', regex=True)\
        .str.replace(' {2,}', ' ', regex=True)\
        .str.strip().replace(r'http\S+', '', regex=True)\
        .replace(r'www\S+', '', regex=True)
    return df

def sentiment_analysis(cloud_df: pd.DataFrame):
    """
    Sentiment analysis score

    Attributes
    ----------
    Dataframe (pandas DF)

    Retrurns
    --------
    Matplotlib Figure
    """
    # print("---------------------------------------------------------------")
    # print(cloud_df)
    # print("---------------------------------------------------------------")
    cloud_df['sentiment'] = cloud_df.message.apply(lambda text: TextBlob(text).sentiment.polarity)
    # print(cloud_df)
    # print("---------------------------------------------------------------")
    sentiment = cloud_df[['name', 'sentiment']].groupby('name').mean()
    # print(sentiment)
    # print("---------------------------------------------------------------")
    s_a = sentiment.sort_values('sentiment', ascending=False).head(15)
    # print(s_a)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(s_a.index, s_a.sentiment, color='orange')
    ax.set_title('Analysis of Sentiment Analysis Score')
    ax.set_xlabel('Name of Group Member')
    ax.set_ylabel('Sentiment Analysis Score')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i, v in enumerate(s_a.sentiment):
        ax.text(i, v, str(v)[0:4], color='black', fontweight='bold', fontsize=9, ha='center', va='bottom', rotation=45, bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.5))
    
    for i, bar in enumerate(ax.patches):
        bar.set_facecolor(np.random.rand(3,))
        bar.set_edgecolor('black')
        bar.set_linewidth(1)
        bar.set_alpha(0.5)
    return fig

def create_wordcloud(df):
    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U0001F1F2-\U0001F1F4"  # Macau flag
            u"\U0001F1E6-\U0001F1FF"  # flags
            u"\U0001F600-\U0001F64F"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U0001F1F2"
            u"\U0001F1F4"
            u"\U0001F620"
            u"\u200d"
            u"\u2640-\u2642"
            "]+", flags=re.UNICODE)
    df['Message'] = df.loc[:, 'Message'].apply(
        lambda s: s.lower())\
        .apply(lambda s: emoji_pattern.sub(r'', s))\
        .str.replace('\n|\t', '', regex=True)\
        .str.replace(' {2,}', ' ', regex=True)\
        .str.strip().replace(r'http\S+', '', regex=True)\
        .replace(r'www\S+', '', regex=True)

    df = df[df['Message'].str.len() > 2]
    df = df[df['User'] != 'group_notification']
    df = df[df['Message'] != '<media omitted>']

    text = " ".join(review for review in df.Message)
    return text
