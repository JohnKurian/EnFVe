import {Input} from "antd";
import React, { useState, useEffect } from "react";
import WordCloud  from "./twitter/WordCloud.jsx";
import NewsArticles from './twitter/NewsArticles';
import TweetCard from './twitter/TweetCard';
import Influencer from './twitter/Influencer.js'
import ReportEditor from './twitter/ReportEditor';

import { Typography } from 'antd';
import TrendGraph from "./twitter/TrendGraph";

import tweets from './twitter/data/tweets'
import hashtags from "./twitter/data/hashtags_wordcloud";
import articles from "./twitter/data/articles";
import influencers from "./twitter/data/influencers";

const { Title } = Typography;

const { TextArea } = Input;



class TwitterDashboard extends React.Component {
    state = {
        value: '',
    };

    onChange = ({ target: { value } }) => {
        this.setState({ value });
    };

    render() {
        const { value } = this.state;

        return (
            <div>
                <div style={{'display': 'flex', 'flexDirection': 'row'}}>
                    <div>
                    <Title>#insurance</Title>
                        <ReportEditor/>
                        <div><TrendGraph/></div>
                    </div>
                    <div><WordCloud hashtags={hashtags}/></div>
                </div>

                <div style={{'display': 'flex', 'flexDirection': 'row'}}>
                    <div style={{'display': 'flex', 'flexDirection': 'column'}}>
                        {tweets.map((tweet,i) => <TweetCard key={i} tweet={tweet}/>)
                        }

                    </div>
                    <div>
                        {influencers.map((influencer,i) => <Influencer key={i} influencer={influencer}/>)
                        }
                    </div>
                </div>


                <div style={{'display': 'flex', 'flexDirection': 'column'}}>
                        <div><NewsArticles articles={articles}/></div>
                </div>

                <div>
                    <div>plotly graph</div>
                </div>


            </div>
    );
    }
}

export default TwitterDashboard;