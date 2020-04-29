import React, { useState, useEffect } from "react";
import io from "socket.io-client";
import {DatePicker, Tooltip, Button, Spin, Card, Avatar} from 'antd';
import { SearchOutlined, QuestionCircleTwoTone, CheckCircleTwoTone, CloseCircleTwoTone } from '@ant-design/icons';
import Demo from './Demo';
import Evidences from './Evidences';
import ResultsGraph from './ResultsGraph'
import { Row, Col } from 'antd';
import { Typography } from 'antd';
import { Progress } from "antd";

import { Input } from 'antd';
import Insights from "./Insights";
import Questions from "./Questions";




const { Search } = Input;

const { Title } = Typography;



const suffix = (
    <SearchOutlined
        style={{
          fontSize: 16,
          color: '#1890ff',
          paddingRight: 4,
        }}
    />
);

let predKey = {
  'True': <CheckCircleTwoTone twoToneColor="#66BB6A" />,
  'Refutes': <CloseCircleTwoTone twoToneColor="#FF7043" />,
  'Not enough Info': <QuestionCircleTwoTone twoToneColor="#FFEE58" />
}

let endPoint = "http://localhost:5000";
let socket = io.connect(`${endPoint}`);

const App = () => {
  const [messages, setMessages] = useState({});
  const [message, setMessage] = useState("");
  const [spinner, setSpinner] = useState(false)
  const [testMessage, setTestMessage] = useState("");

  useEffect(() => {
    getMessages();
  }, [messages.length]);

  const getMessages = () => {
    socket.on("message", msg => {
      //   let allMessages = messages;
      //   allMessages.push(msg);
      //   setMessages(allMessages);
      setSpinner(false)
      setMessages(msg);
    });
  };

  // On Change
  const onChange = e => {
    setMessage(e.target.value);
  };

  // On Click
  const onClick = () => {
    if (message !== "") {
      setSpinner(true)
      socket.emit("message", message);

      fetch('/')
          .then(res => {console.log('inside fetch'); setTestMessage(res.json());})
          .then(
              (result) => {
                console.log('inside fetch:', result)
                setTestMessage(result);
              },
              // error handler
              (error) => {
                // this.setState({
                //   isLoaded: true,
                //   error
                // });
              }
          )
      setMessage("");
    } else {
      alert("Please Add A Message");
    }
  };

  const onCustomSearchClick = (message) => {
    if (message !== "") {
      setSpinner(true)
      socket.emit("message", message);
      setMessage("");
    } else {
      alert("Please Add A Message");
    }
  };

  console.log('messages:', messages)

  return (

      <div>
    <div style={{'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'}}>
        <div style={{'display': 'flex', 'flex-direction': 'row', 'align-items': 'center'}}>
        <img src={ require('./icons8-james-bond-64.png') } />
        {/*<Spin />*/}
        <Title level={2}>Detective AI: Explainable fact verification</Title>
        </div>
        <div style={{'display': 'flex', 'flex-direction': 'row', 'align-items': 'center'}}>

          {/*<input value={message} name="message" onChange={e => onChange(e)} />*/}
          {/*/!*<button onClick={() => onClick()}>Fact Check</button>*!/*/}

          {/*<Button type="primary" icon={<SearchOutlined />} onClick={() => onClick()}>*/}
          {/*  Check*/}
          {/*</Button>*/}


          <Search
              placeholder="Enter a fact to check"
              enterButton="Check"
              size="large"
              style={{ width: 600 }}
              suffix={suffix}
              onSearch={message => onCustomSearchClick(message)}
          />

            {testMessage}

      {/*<Demo/>*/}
      {/*  <Tooltip title="search">*/}
      {/*      <Button type="primary" shape="circle" icon={<SearchOutlined />} />*/}
      {/*  </Tooltip>*/}
        </div>


      <br/>
      <br/>
      { spinner && <Spin size="large" /> }

      { spinner !==true && Object.keys(messages).length === 0 &&
        <div style={{'display': 'flex', 'flex-direction': 'row', 'align-items': 'flex-start'}}>
          <div style={{'margin': '10px'}}>
            <Card title="Default size card" extra={<a href="#">More</a>} style={{ width: 300 }}>
              <p>Card content</p>
              <p>Card content</p>
              <p>Card content</p>
            </Card>
          </div>
          <div style={{'margin': '10px'}}>
            <Card title="Default size card" extra={<a href="#">More</a>} style={{ width: 300 }}>
              <p>Card content</p>
              <p>Card content</p>
              <p>Card content</p>
            </Card>
          </div>
          <div style={{'margin': '10px'}}>
            <Card title="Default size card" extra={<a href="#">More</a>} style={{ width: 300 }}>
              <p>Card content</p>
              <p>Card content</p>
              <p>Card content</p>
            </Card>
          </div>
        </div>
      }




        { Object.keys(messages).length > 0 && spinner !==true &&
        messages['gear_results'].map(message => <div style={{'display': 'flex', 'flex-direction': 'row', 'align-items': 'flex-start'}}>
            <div style={{'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'}}>
              <div> {predKey[message.prediction_result]} {message.prediction_result}</div>
              <ResultsGraph evidences={message}/>
                <Insights messages={messages}/>
            </div>
            <Evidences evidences={message}/>
            <Questions messages={messages}/>
          </div>)
        }


{/*       { Object.keys(messages).length > 0 && spinner !==true && */}

{/*       } */}


    </div>
      </div>
  );
};

export default App;
