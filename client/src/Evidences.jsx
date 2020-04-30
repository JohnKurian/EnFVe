import React, { useState, useEffect } from "react";
import { Row, Col, Slider } from 'antd';

import { Card, Avatar } from 'antd';

const { Meta } = Card;

class Evidences extends React.Component {

    gutters = {};

    vgutters = {};

    colCounts = {};

    constructor() {
        super();
        this.state = {
            gutterKey: 1,
            vgutterKey: 1,
            colCountKey: 2,
            text: 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum'
        };
        [8, 16, 24, 32, 40, 48].forEach((value, i) => {
            this.gutters[i] = value;
        });
        [8, 16, 24, 32, 40, 48].forEach((value, i) => {
            this.vgutters[i] = value;
        });
        [2, 3, 4, 6, 8, 12].forEach((value, i) => {
            this.colCounts[i] = value;
        });
    }

    onGutterChange = gutterKey => {
        this.setState({ gutterKey });
    };

    onVGutterChange = vgutterKey => {
        this.setState({ vgutterKey });
    };

    onColCountChange = colCountKey => {
        this.setState({ colCountKey });
    };

    onLogoClick = click => {
        console.log('on logo click')
    }

    render() {
        const { gutterKey, vgutterKey, colCountKey, text } = this.state;
        const cols = [];
        const colCount = this.colCounts[colCountKey];
        let colCode = '';
        for (let i = 0; i < colCount; i++) {
            cols.push(
                <Col key={i.toString()} span={24 / colCount}>
                    <div>Column</div>
                </Col>,
            );
            colCode += `  <Col span={${24 / colCount}} />\n`;
        }

        const evidences = []

        if(this.props.evidences!==undefined) {
            for (let i = 0; i < 5; i++) {
                let link = "https://en.wikipedia.org/wiki/" + this.props.evidences.wiki_results[i]
                evidences.push(
                        <Card style={{ width: 500 }}>

                            <Meta
                                avatar={<Avatar src={this.props.evidences.img_urls[i]} />}
                                title={this.props.evidences.wiki_results[i]}
                            />

                            <p><i>{this.props.evidences.paras_joined[i]}</i></p>
                            <img style={{width: 30, height: 30, padding: '5px'}}
                                 src={require('./icons8-wikipedia-50.png')}></img>
                            <a href={link} target="_blank">
                                read more
                            </a>

                        </Card>
                )
            }
        }
        return (

                    <div style={{'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'}}>
                        {evidences}
                        {/*{this.props.evidences!==undefined && this.props.evidences.length > 0 &&*/}
                        {/*this.props.evidences.paras.map(msg => (*/}
                        {/*    <div>*/}
                        {/*        <p>{msg}</p>*/}
                        {/*    </div>*/}
                        {/*))}*/}
                        {/*}*/}
                    </div>

        );
    }
}

export default Evidences;