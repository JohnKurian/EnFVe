import {Input} from "antd";
import React, { useState, useEffect } from "react";
import WordCloud  from "./twitter/WordCloud.jsx";
import NewsArticles from './twitter/NewsArticles';
import TweetCard from './twitter/TweetCard';
import Influencer from './twitter/Influencer.js'


import { QuestionCircleTwoTone } from '@ant-design/icons';
import {NavLink as RouterNavLink} from "react-router-dom";
import {NavLink} from "reactstrap";



import {
    BrowserRouter as Router,
    Link,
    Route // for later
} from 'react-router-dom'

import config from "../auth_config.json";


// import ReportEditor from './twitter/ReportEditor';

import { Typography } from 'antd';
import TrendGraph from "./twitter/TrendGraph";

import tweets from './twitter/data/tweets'
import hashtags from "./twitter/data/hashtags_wordcloud";
import articles from "./twitter/data/articles";
import influencers from "./twitter/data/influencers";
import factcheckresults from "./twitter/data/factcheckresults";
// import community_graph from "./twitter/data/community_graph";
import graph_img from './twitter/data/graph.png';

// import LandingMenu from "./Menu";
import Topheader from "./Header";
import {useAuth0} from "../react-auth0-spa";

const { apiOrigin = "http://localhost:3001" } = config;

const axios = require('axios')

const { Title } = Typography;

const { TextArea } = Input;

// //Your React component
// fetchExternalHTML: function(fileName) {
//     Ajax.getJSON('/myAPI/getExternalHTML/' + fileName).then(
//         response => {
//             this.setState({
//                 extHTML: response
//             });
//         }, err => {
//             //handle your error here
//         }
//     );
// }



const TwitterDashboard = (props) => {

    const { match: { params } } = props;

    const { getTokenSilently, user } = useAuth0();

    const [report, setReport] = useState(undefined);


    const callApi = async (params) => {
        try {
            let report_id = params['report']
            const token = await getTokenSilently();

            const response = await fetch(`${apiOrigin}/api/get_report/`, {
                method: 'POST',
                headers: {
                    Authorization: `Bearer ${token}`,
                    Accept: 'application/json',
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    'report_id': report_id
                })
            });

            const responseData = await response.json();
            console.log('server response:', responseData['report'])
            setReport(responseData['report']);
        } catch (error) {
            console.error(error);
        }
    };

    useEffect(() => {
        console.log('inside twitter dashboard:', params)
        callApi(params)
    }, [params]);

    return (
        <div>
        {report && <div>
            <Title>#insurance</Title>
            <div style={{'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-around'}}>
                <div>
                    {/*<ReportEditor/>*/}
                    {/*<div><TrendGraph/></div>*/}
                </div>
                <div><WordCloud hashtags={report.hashtag_wordclouds}/></div>
            </div>

            <div style={{'display': 'flex', 'flexDirection': 'row', 'background': '#fdfdfd', 'padding': '50px'}}>
                <div>
                    <div style={{'margin': '20px', 'fontSize': '23px'}}>
                        Most viral tweets
                    </div>
                    <div style={{
                        'display': 'flex',
                        'flexDirection': 'column',
                        'marginRight': '85px',
                        'height': '750px',
                        'overflow': 'hidden',
                        'overflowY': 'scroll'
                    }}>
                        {report.viral_tweets.map((tweet, i) => <TweetCard key={i} tweet={tweet} factcheckresults={factcheckresults}/>)
                        }

                    </div>
                </div>
                <div>
                    <div style={{'margin': '20px', 'fontSize': '23px'}}>
                        Key influencers
                    </div>
                    <div style={{
                        'display': 'flex',
                        'flexDirection': 'column',
                        'marginRight': '200px',
                        'height': '750px',
                        'overflow': 'hidden',
                        'overflowY': 'scroll'
                    }}>
                        {report.influencers.map((influencer, i) => <Influencer key={i} influencer={influencer}/>)
                        }
                    </div>
                </div>
            </div>


            <div style={{'display': 'flex', 'justifyContent': 'center'}}>

                <div style={{'width': '900px', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}}>
                    <div style={{'margin': '20px', 'fontSize': '23px'}}>Most shared articles</div>
                    <div style={{'display': 'flex', 'flexDirection': 'row', 'width': '1000px', 'overflowX': 'auto'}}>
                        <NewsArticles articles={report.news_articles}/>
                    </div>
                </div>
            </div>

            <div style={{'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}}>
                <div style={{'margin': '20px', 'fontSize': '23px'}}>Community graph</div>
                <img style={{'width': '600px'}} src={graph_img}/>
            </div>


        </div>
}
</div>
    );

}

export default TwitterDashboard;