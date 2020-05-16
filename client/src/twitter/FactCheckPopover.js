import { Popover, Button } from 'antd';
import {Input} from "antd";
import React, { useState, useEffect } from "react";
import App from "../App";
import ResultsGraph from "../ResultsGraph";
import Insights from "../Insights";
import Evidences from "../Evidences";
import { SearchOutlined, QuestionCircleTwoTone, CheckCircleTwoTone, CloseCircleTwoTone } from '@ant-design/icons';


const { TextArea } = Input;

let predKey = {
    'True': <CheckCircleTwoTone twoToneColor="#66BB6A" />,
    'Refutes': <CloseCircleTwoTone twoToneColor="#FF7043" />,
    'Not enough Info': <QuestionCircleTwoTone twoToneColor="#FFEE58" />
}


let predKeyColor = {
    'True': '#66BB6A',
    'Refutes': "#FF7043",
    'Not enough Info': "#FFEE58"
}





// { Object.keys(messages).length > 0 && spinner !==true &&
// messages['gear_results'].map(message => <div style={{'display': 'flex', 'flex-direction': 'row', 'align-items': 'flex-start'}}>
//     <div style={{'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'}}>
//         <div style={{'fontSize': '25px'}}> {predKey[message.prediction_result]} {message.prediction_result}</div>
//         <ResultsGraph evidences={message}/>
//         <Insights messages={messages}/>
//     </div>
//     <Evidences evidences={message}/>
//     {/*             <Questions messages={messages}/> */}
// </div>)
// }


class FactCheckResults extends React.Component {
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
                { Object.keys(this.props.factcheckresults).length > 0 &&
                this.props.factcheckresults['gear_results'].map(message => <div style={{'display': 'flex', 'flex-direction': 'row', 'align-items': 'flex-start'}}>
                    <div style={{'display': 'flex', 'flex-direction': 'row', 'align-items': 'center'}}>
                        <div style={{'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'}}>
                        <div style={{'fontSize': '25px'}}> {predKey[message.prediction_result]} {message.prediction_result}</div>
                        <ResultsGraph evidences={message}/>
                        </div>
                        <Evidences style={{'width': '100px', 'fontSize': '12px'}} evidences={message}/>
                    </div>
                    {/*             <Questions messages={messages}/> */}
                </div>)
                }
            </div>
        );
    }
}






class FactCheckPopover extends React.Component {
    state = {
        value: '',
    };

    onChange = ({ target: { value } }) => {
        this.setState({ value });
    };



    render() {
        console.log(this.props.factcheckresults)
        const { value } = this.state;

        return (
            <Popover content={<FactCheckResults factcheckresults={this.props.factcheckresults}/>} title="Fact check results">
                <p style={{'color': predKeyColor[this.props.factcheckresults['gear_results'][0]['prediction_result']]}}>fact-checked {predKey[this.props.factcheckresults['gear_results'][0]['prediction_result']]} </p>
            </Popover>
        );
    }
}

export default FactCheckPopover;