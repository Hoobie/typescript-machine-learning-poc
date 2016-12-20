/// <reference path='declarations.d.ts'/>

import * as MlKnn from 'ml-knn';
import * as KNear from 'knear';
import * as RF from 'random-forest-classifier';
import * as MlNaiveBayes from 'ml-naivebayes';
import * as QLearning from 'q-exp';

var data = JSON.parse(document.getElementById('data').textContent);

testMlKnn();
testKNear();
testRandomForestClassifier();
testMlNaiveBayes();
testQLearning();

function testMlKnn() {
    let knn = new MlKnn();

    knn.train(data.trainingSet, data.trainingPredictions);

    let predictions = knn.predict(data.testSet);

    console.log('ml-knn predictions: ', predictions);
    console.log('equal to test predictions? ', JSON.stringify(predictions) === JSON.stringify(data.testPredictions));
}

function testKNear() {
    var k = 3;
    var machine = new KNear.kNear(k);

    for (var i = 0; i < data.trainingSet.length; i++) {
        machine.learn(data.trainingSet[i], data.trainingPredictions[i]);
    }

    var predictions = [];
    for (var i = 0; i < data.testSet.length; i++) {
        predictions.push(machine.classify(data.testSet[i]));
    }
    console.log('knear predictions: ', predictions);
    console.log('equal to test predictions? ', JSON.stringify(predictions) === JSON.stringify(data.testPredictions));
}

function testRandomForestClassifier() {
    var trainingSet = [];
    for (var i = 0; i < data.trainingSet.length; i++) {
        trainingSet.push({
            "lines_count": data.trainingSet[i][0],
            "parameters_count": data.trainingSet[i][1],
            "parameters_size": data.trainingSet[i][2],
            "execution": data.trainingPredictions[i]
        });
    }
    var testSet = [];
    for (var i = 0; i < data.testSet.length; i++) {
        testSet.push({
            "lines_count": data.testSet[i][0],
            "parameters_count": data.testSet[i][1],
            "parameters_size": data.testSet[i][2]
        });
    }

    var rf = new RF.RandomForestClassifier({
        n_estimators: 10
    });

    rf.fit(trainingSet, null, "execution", function(err, trees) {
        var predictions = rf.predict(testSet, trees);
        console.log('random-forest-classifier predictions: ', predictions);
        console.log('equal to test predictions? ', JSON.stringify(predictions) === JSON.stringify(data.testPredictions));
    });
}

function testMlNaiveBayes() {
    let nb = new MlNaiveBayes();

    nb.train(data.trainingSet, data.trainingPredictions);

    let predictions = nb.predict(data.testSet);

    console.log('ml-naivebayes predictions: ', predictions);
    console.log('equal to test predictions? ', JSON.stringify(predictions) === JSON.stringify(data.testPredictions));
}

function testQLearning() {
    let ql = QLearning;

    var agent = ql
        .newAgent('johndoe', ['walk', 'run', 'sleep'], 0.35)
        .then(ql.bindRewardMeasure(state => 10))
        .then(ql.bindActionCostMeasure((state, action) => 10))
        .then(ql.bindStateGenerator((state, action) => 'walk'));
        //.then(ql.load('???'));

    // Start!
    agent.then(ql.setState('walk')) // Let the agent know the state
        .then(ql.step) // Ask the agent to move
        .then(ql.getState) // Now let's see how the agent moved
        .then((state) => console.log(state));

   console.log(agent);
}
