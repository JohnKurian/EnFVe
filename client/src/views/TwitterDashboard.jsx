import {Input} from "antd";
import React, { useState, useEffect } from "react";
import WordCloud  from "./twitter/WordCloud.jsx";
import NewsArticles from './twitter/NewsArticles';
import TweetCard from './twitter/TweetCard';
import Influencer from './twitter/Influencer.js'
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



class TwitterDashboard extends React.Component {
    state = {
        value: '',
    };

    onChange = ({ target: { value } }) => {
        this.setState({ value });
    };

    // createMarkup() {
    //     return {__html: community_graph};
    // }

    render() {
        const { value } = this.state;

        return (
            <div>
                <Title>#Plandemic</Title>
                <div style={{'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-around'}}>
                    <div>
                        {/*<ReportEditor/>*/}
                        {/*<div><TrendGraph/></div>*/}
                    </div>
                    <div><WordCloud hashtags={hashtags}/></div>
                </div>

                <div style={{'display': 'flex', 'flexDirection': 'row',  'background': '#fdfdfd', 'padding': '50px'}}>
                    <div>
                        <div style={{'margin': '20px', 'fontSize': '23px'}}>
                            Most viral tweets
                        </div>
                        <div style={{'display': 'flex', 'flexDirection': 'column', 'marginRight': '85px', 'height': '750px', 'overflow': 'hidden', 'overflowY': 'scroll'}}>
                            {tweets.map((tweet,i) => <TweetCard key={i} tweet={tweet} factcheckresults={factcheckresults}/>)
                            }

                        </div>
                    </div>
                    <div>
                        <div style={{'margin': '20px', 'fontSize': '23px'}}>
                            Key influencers
                        </div>
                        <div style={{'display': 'flex', 'flexDirection': 'column', 'marginRight': '200px', 'height': '750px', 'overflow': 'hidden', 'overflowY': 'scroll'}}>
                            {influencers.map((influencer,i) => <Influencer key={i} influencer={influencer}/>)
                            }
                        </div>
                    </div>
                </div>


                <div style={{'display': 'flex',  'justifyContent': 'center'}}>

                        <div style={{'width': '900px', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}}>
                            <div style={{'margin': '20px', 'fontSize': '23px'}}>Most shared articles</div>
                            <div style={{'display': 'flex', 'flexDirection': 'row', 'width': '1000px', 'overflowX': 'auto'}}>
                                <NewsArticles articles={articles}/>
                            </div>
                        </div>
                </div>

                <div style={{'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}}>
                    <div style={{'margin': '20px', 'fontSize': '23px'}}>Community graph</div>
                    <img style={{'width': '600px'}} src={graph_img} />
                </div>


            </div>
    );
    }
}

export default TwitterDashboard;