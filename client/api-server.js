const express = require("express");
const cors = require("cors");
const morgan = require("morgan");
const helmet = require("helmet");
const jwt = require("express-jwt");
const jwksRsa = require("jwks-rsa");
const authConfig = require("./src/auth_config.json");
var bodyParser = require('body-parser')
const axios = require('axios')

const app = express();

const port = process.env.API_PORT || 3001;
const appPort = process.env.SERVER_PORT || 3000;
const appOrigin = authConfig.appOrigin || `http://127.0.0.1:${appPort}`;

if (!authConfig.domain || !authConfig.audience) {
  throw new Error(
    "Please make sure that auth_config.json is in place and populated"
  );
}

app.use(morgan("dev"));
app.use(helmet());
app.use(cors({ origin: appOrigin }));
app.use(bodyParser());

const checkJwt = jwt({
  secret: jwksRsa.expressJwtSecret({
    cache: true,
    rateLimit: true,
    jwksRequestsPerMinute: 5,
    jwksUri: `https://${authConfig.domain}/.well-known/jwks.json`
  }),

  audience: authConfig.audience,
  issuer: `https://${authConfig.domain}/`,
  algorithm: ["RS256"]
});

app.post("/api/external", checkJwt, (req, res) => {
  console.log(req.user)
  console.log(req.body)
  res.send({
    msg: "Your access token was successfully validated!"
  });
});


app.post("/api/get_reports", checkJwt, (req, res) => {
  console.log(req.user)
  console.log(req.body)
  axios.post('http://0.0.0.0:16000/get_reports', {
    hashtags: req.body.hashtags,
    user: req.body.user
  })
      .then((resp) => {
        console.log(`statusCode: ${resp.statusCode}`)
        res.send({
            reports: resp['report_list']
          });

        // console.log(res)
      })
      .catch((error) => {
        console.error(error)
      })



  res.send({
    msg: "Your access token was successfully validated!"
  });
});



app.post("/api/createreport", checkJwt, (req, res) => {
  console.log(req.body)
  const axios = require('axios')

  axios.post('http://0.0.0.0:16000/', {
    hashtags: req.body.hashtags,
    user: req.body.user
  })
      .then((res) => {
        console.log(`statusCode: ${res.statusCode}`)
        // console.log(res)
      })
      .catch((error) => {
        console.error(error)
      })


  axios.post('http://0.0.0.0:16000/update_user', {
    user: req.body.user
  })
      .then((res) => {
        console.log(`statusCode: ${res.statusCode}`)
        // console.log(res)
      })
      .catch((error) => {
        console.error(error)
      })

  res.send({
    msg: "report has been successfully created!"
  });
});



app.listen(port, () => console.log(`API Server listening on port ${port}`));
