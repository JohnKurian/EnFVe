import { List, Avatar, Space } from 'antd';
import { MessageOutlined, LikeOutlined, StarOutlined } from '@ant-design/icons';
import React from "react";

import { Card } from 'antd';
import { EditOutlined, EllipsisOutlined, SettingOutlined } from '@ant-design/icons';

const listData = [];
for (let i = 0; i < 23; i++) {
    listData.push({
        href: 'http://ant.design',
        title: `ant design part ${i}`,
        avatar: 'https://zos.alipayobjects.com/rmsportal/ODTLcjxAfvqbxHnVXCYX.png',
        description:
            'Ant Design, a design language for background applications, is refined by Ant UED Team.',
        content:
            'We supply a series of design principles, practical patterns and high quality design resources (Sketch and Axure), to help people create their product prototypes beautifully and efficiently.',
    });
}

const IconText = ({ icon, text }) => (
    <Space>
        {React.createElement(icon)}
        {text}
    </Space>
);

const { Meta } = Card;








class NewsArticles extends React.Component{

    render() {
        return (
                this.props.articles.map((article, index) => (
            <Card
                onClick={()=> window.open(article.url, "_blank")}
                hoverable
                style={{ width: 400, 'marginRight': '10px' }}
                cover={
                    <img
                        alt="example"
                        height='200px'
                        width='100%'
                        src={article.image}
                    />
                }
                actions={[
                    <SettingOutlined key="setting" />,
                    <EditOutlined key="edit" />,
                    <EllipsisOutlined key="ellipsis" />,
                ]}
            >
                <Meta
                    avatar={<Avatar src={article.favicon} />}
                    title={article.title}
                    description={article.description.substring(0, 100)}
                />
            </Card>
                ))
        )
    }

}
//
// class NewsArticles extends React.Component {
//
//     render() {
//         return (
//             <List
//                 itemLayout="vertical"
//                 size="large"
//                 pagination={{
//                     onChange: page => {
//                         console.log(page);
//                     },
//                     pageSize: 5,
//                 }}
//                 dataSource={this.props.articles}
//                 renderItem={item => (
//                     <List.Item
//                         key={item.title}
//                         actions={[
//                             <IconText icon={StarOutlined} text={item.share_count} key="list-vertical-star-o"/>
//                         ]}
//                         extra={
//                             <img
//                                 width={272}
//                                 alt="logo"
//                                 src={item.image}
//                             />
//                         }
//                     >
//                         <List.Item.Meta
//                             avatar={<Avatar src={item.favicon}/>}
//                             title={<a href={item.href}>{item.title}</a>}
//                             description={item.description}
//                         />
//                         {item.content}
//                     </List.Item>
//                 )}
//             />
//
//         )
//     }
// }

export default NewsArticles;