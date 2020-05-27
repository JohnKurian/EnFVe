import React from "react";
import { Router, Route, Switch } from "react-router-dom";
import { Container } from "reactstrap";

import PrivateRoute from "./components/PrivateRoute";
import Loading from "./components/Loading";
import NavBar from "./components/NavBar";
import Footer from "./components/Footer";
import Home from "./views/Home";
import Profile from "./views/Profile";
import ExternalApi from "./views/ExternalApi";
import Reports from "./views/Reports";
import { useAuth0 } from "./react-auth0-spa";
import history from "./utils/history";
import Report from "./views/Report";
import CreateReport from "./views/CreateReport";

// styles
import "./App.css";

// fontawesome
import initFontAwesome from "./utils/initFontAwesome";
import TwitterDashboard from "./views/TwitterDashboard";
initFontAwesome();

const App = () => {
  const { loading } = useAuth0();

  if (loading) {
    return <Loading />;
  }

  return (
    <Router history={history}>
      <div id="app" className="d-flex flex-column h-100">
        <NavBar />
        <Container className="flex-grow-1 mt-5">
          <Switch>
            <Route path="/" exact component={Home} />
            <PrivateRoute path="/profile" exact component={Profile} />
            <PrivateRoute path="/external-api" exact component={ExternalApi} />
            <PrivateRoute path="/reports" exact component={Reports} />
            <PrivateRoute path="/reports/:report" exact component={TwitterDashboard} />
            <PrivateRoute path="/createreport" exact component={CreateReport} />
          </Switch>
        </Container>
        <Footer />
      </div>
    </Router>
  );
};

export default App;
