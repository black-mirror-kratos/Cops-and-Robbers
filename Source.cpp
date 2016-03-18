#include <iostream>
#include <queue>
#include <vector>
#include <functional>
#include <ctime>
#include <random>
#include <stdlib.h>
#include <windows.h>
#include <cassert>
#include "Net.h"

using namespace std;

#define PII pair<int,int>
#define drawVector vector<pair<PII*,char>>
#define MP make_pair
#define rep(i,n) for(unsigned i=0;i<n;i++)
#define reep(i,a,b) for(unsigned i=a;i<=b;i++)
#define per(i,n) for(unsigned i=0;i>n;i--)
#define peer(i,a,b) for(unsigned i=a;i>=b;i--)
#define fcin freopen("std.in","r",stdin)
#define fcout freopen("std.out","w",stdout)

int N;
int M;
int frame;
int frameLimit = 40000;
int sleepTime = 0;
int thiefMoves = 1;
int layers = 3;
int numOpponents = 1;

void draw(drawVector &v, int N, int M) {
	if(frame >= frameLimit) Sleep(sleepTime);
	system("CLS");
	vector<vector<char>> board;
	board.resize(N+2, vector<char>(M+2));
	for (int i = 0; i < M+2; i++)
		board[0][i] = '#';
	cout << endl;
	for (int i = 1; i < N+1; i++) {
		board[i][0] = '#';
		for (int j = 1; j < M+1; j++) {
			board[i][j] = '.';
		}
		board[i][M+1] = '#';
	}
	for (int i = 0; i < M+2; i++)
		board[N+1][i] = '#';
	for (auto i : v) {
		PII* loc = i.first;
		char marker = i.second;
		cout << loc->first << "," << loc->second << endl;
		board[loc->first+1][loc->second+1] = marker;
	}
	for (int i = 0; i < N+2; i++) {
		for (int j = 0; j < M+2; j++) {
			cout << board[i][j] << " ";
		}
		cout << endl;
	}
}

int random(int a, int b) {
	std::random_device rd; // obtain a random number from hardware
	std::mt19937 eng(rd()); // seed the generator
	std::uniform_int_distribution<> distr(a, b); // define the range
	return distr(eng);
}

bool isFeasible(PII loc) {
	if (loc.first < 0 || loc.first >= N || loc.second < 0 || loc.second >= M)
		return false;
	return true;
}

void randMove(PII &loc) {
	PII temp;
	do{
		temp = loc;
		temp.first += random(-1, 1);
		temp.second += random(-1, 1);
	} while (!isFeasible(temp));
	loc = temp;
}

double H(PII a, PII b) {
	int deltaX = abs(a.first - b.first);
	int deltaY = abs(a.second - b.second);
	//return deltaX + deltaY;
	return sqrt(deltaX*deltaX + deltaY*deltaY);
}

void neuralMove(PII &player, vector<PII> &opponents, int N, int M, Net &myNet) {
	PII temp;
	temp = player;
	for (auto i : opponents) {
		if (player == i) {
			vector<double> deltas;
			double deltaX = 0.0;
			double deltaY = 0.0;
			do{
				deltaX = random(0, 1) ? 1 : -1;
				deltaY = random(0, 1) ? 1 : -1;
			} while (!isFeasible(MP(temp.first + deltaX, temp.second + deltaY)));
			deltas.push_back(deltaX);
			deltas.push_back(deltaY);
			/*if(outputXE == 0) deltas.push_back(1.0);
			else deltas.push_back(0.0);
			if (outputYE == 0) deltas.push_back(1.0);
			else deltas.push_back(0.0);*/
			myNet.backProp(deltas);
		}
		else
			if (H(player, i) < 1.415) {
				vector<double> deltas;
				double mag = H(player, i) / 42.427;
				double dirX;
				double dirY;
				if (abs(i.first - player.first) != 0) dirX = (i.first - player.first) / (abs(i.first - player.first) + 0.0);
				else dirX = 0;
				double magX = abs(i.first - player.first) / (M + 0.0);
				if (abs(i.second - player.second) != 0) dirY = (i.second - player.second) / (abs(i.second - player.second) + 0.0);
				else dirY = 0;
				double magY = abs(i.second - player.second) / (N + 0.0);
				deltas.push_back(-1 * dirY);
				deltas.push_back(-1 * dirX);
				/*deltas.push_back(0.0);
				deltas.push_back(0.0);*/
				myNet.backProp(deltas);
			}
	}

	double normalizationFactor = max(N, M) + 0.0;
	vector<double> input;
	input.push_back(player.first/normalizationFactor);
	input.push_back(player.second / normalizationFactor);
	for (auto i : opponents) {
		input.push_back(i.first / normalizationFactor);
		input.push_back(i.second / normalizationFactor);
	}
	input.push_back(N / normalizationFactor);
	input.push_back(M / normalizationFactor);
	myNet.feedForward(input);
	vector<double> res;
	myNet.getResults(res);
	double outputX = res[0];
	double outputY = res[1];
	//double outputXE = res[2];
	//double outputYE = res[3];
	//cout <<"Nueral Output : "<< outputX << "," << outputY << endl;
	if (outputX <= -0.5) outputX = -1;
	else if(outputX >= 0.5) outputX = 1;
	else outputX = 0.0;
	if (outputY <= -0.5) outputY = -1;
	else if (outputY > 0.5) outputY = 1;
	else outputY = 0.0;
	/*if (outputXE <= 0.0) outputXE = 0;
	else if (outputXE > 0.0) outputXE = 1;
	if (outputYE <= 0.0) outputYE = 0;
	else if (outputYE > 0.0) outputYE = 1;*/
	//else outputY = 0;
	//cout << "Modified Nueral Output : " << outputX << "," << outputY << endl;

	temp = player;
	/*if(outputXE == 1) */temp.second += outputX;
	/*if(outputYE == 1) */temp.first += outputY;
	if (!isFeasible(temp)) {
		vector<double> deltas;
		if (temp.second<0) deltas.push_back(1.0);
		else if (temp.second >= M) deltas.push_back(-1.0);
		else deltas.push_back(0.0);
		if (temp.first<0) deltas.push_back(1.0);
		else if (temp.first >= N) deltas.push_back(-1.0);
		else deltas.push_back(0.0);
		//cout << "Correcting Deltas : " << deltas[0]<< "," << deltas[1] << endl;
		/*deltas.push_back(0.0);
		deltas.push_back(0.0);*/
		myNet.backProp(deltas);
		return;
	}
	player = temp;
	for (auto i : opponents) {
		if (player == i) {
			vector<double> deltas;
			if (isFeasible(MP(temp.first + 1, temp.second + 1))) {
				deltas.push_back(1);
				deltas.push_back(1);
			}
			else if (isFeasible(MP(temp.first - 1, temp.second - 1))) {
				deltas.push_back(-1);
				deltas.push_back(-1);
			}
			else if (isFeasible(MP(temp.first - 1, temp.second + 1))) {
				deltas.push_back(1);
				deltas.push_back(-1);
			}
			else if (isFeasible(MP(temp.first + 1, temp.second - 1))) {
				deltas.push_back(-1);
				deltas.push_back(1);
			}
			/*if(outputXE == 0) deltas.push_back(1.0);
			else deltas.push_back(0.0);
			if (outputYE == 0) deltas.push_back(1.0);
			else deltas.push_back(0.0);*/
			myNet.backProp(deltas);
		}
		else
			if (H(player, i) < 1.415) {
				vector<double> deltas;
				double mag = H(player, i) / 42.427;
				double dirX;
				double dirY;
				if (abs(i.first - player.first) != 0) dirX = (i.first - player.first) / (abs(i.first - player.first) + 0.0);
				else dirX = 0;
				double magX = abs(i.first - player.first) / (M + 0.0);
				if (abs(i.second - player.second) != 0) dirY = (i.second - player.second) / (abs(i.second - player.second) + 0.0);
				else dirY = 0;
				double magY = abs(i.second - player.second) / (N + 0.0);
				deltas.push_back(-1 * dirY);
				deltas.push_back(-1 * dirX);
				/*deltas.push_back(0.0);
				deltas.push_back(0.0);*/
				myNet.backProp(deltas);
			}
	}
}


void seeAndMove(PII &player, PII &opponent) {
	if (player == opponent) return;
	double dirX;
	double dirY;
	double magY = abs(opponent.first - player.first) + 0.0;
	double magX = abs(opponent.second - player.second) + 0.0;
	if (abs(opponent.first - player.first) != 0) dirY = (opponent.first - player.first) / (abs(opponent.first - player.first) + 0.0);
	else dirY = 0;
	if (abs(opponent.second - player.second) != 0)dirX = (opponent.second - player.second) / (abs(opponent.second - player.second) + 0.0);
	else dirX = 0;
	if (abs(dirX) == 1 && abs(dirY) == 1) {
		if (magX > magY) dirY = 0;
		else dirX = 0;
	}
	//cout << "see and move : " << dirX<< "," << dirY << endl;
	player.first += dirY;
	player.second += dirX;
}

int main() {
	fcin;
	//fcout;
	vector<unsigned> topology;
	unsigned t;
	for (int i = 0;i < layers;i++) {
		cin >> t;topology.push_back(t);
	}
	Net myNet(topology);

	N = 20;
	M = 20;
	
	vector<PII> opponents;
	for (int i = 0;i < numOpponents; i++) {
		opponents.push_back(MP(random(0, N - 1), random(0, M - 1)));
	}
	//PII thief = MP(opponents[0].first-1, opponents[0].second-1);
	PII thief = MP(random(0, N - 1), random(0, M - 1));

	drawVector D;
	D.push_back(MP(&thief, 'o'));
	for (int i = 0;i < numOpponents; i++) {
		D.push_back(MP(&opponents[i], 'x'));
	}

	draw(D, N, M);

	frame = 0;

	while(true) {
		for (int i = 0;i < numOpponents; i++) {
			seeAndMove(opponents[i], thief);
			if (frame >= frameLimit) draw(D, N, M);
		}
		for (int i = 0;i < thiefMoves;i++) {
			neuralMove(thief, opponents, N, M, myNet);
			if (frame >frameLimit) draw(D, N, M);
		}
		
		frame++;
	}

	return 0;
}