import React, { useState, useEffect } from "react";
import io from "socket.io-client";
import { DatePicker, Tooltip, Button, Spin } from 'antd';
import { SearchOutlined, QuestionCircleTwoTone, CheckCircleTwoTone, CloseCircleTwoTone } from '@ant-design/icons';
import Demo from './Demo';
import Evidences from './Evidences';
import ResultsGraph from './ResultsGraph'
import { Row, Col } from 'antd';
import { Typography } from 'antd';

import { Input } from 'antd';




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
  const [messages, setMessages] = useState([]);
  const [message, setMessage] = useState("");
  const [spinner, setSpinner] = useState(false)

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

      {/*<Demo/>*/}
      {/*  <Tooltip title="search">*/}
      {/*      <Button type="primary" shape="circle" icon={<SearchOutlined />} />*/}
      {/*  </Tooltip>*/}
        </div>


      <br/>
      <br/>
      { spinner && <Spin size="large" /> }



        { Object.keys(messages).length > 0 && spinner !==true &&
        <div style={{'display': 'flex', 'flex-direction': 'row', 'align-items': 'flex-start'}}>
          <div style={{'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'}}>
            <div> {predKey[messages.prediction_result]} {messages.prediction_result}</div>
        <ResultsGraph evidences={messages}/>
          </div>
          <Evidences evidences={messages}/>
        </div>
        }


    </div>
      </div>
  );
};

export default App;
