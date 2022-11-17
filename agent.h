/**
 * Framework for NoGo and similar games (C++ 11)
 * agent.h: Define the behavior of variants of the player
 *
 * Author: Theory of Computer Games
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include <fstream>
#include "board.h"
#include "action.h"

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string property(const std::string& key) const { return meta.at(key); }
	virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }

protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
};

/**
 * base agent for agents with randomness
 */
class random_agent : public agent {
public:
	random_agent(const std::string& args = "") : agent(args) {
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

class Node {
public:
	int total = 0;
	int win = 0;
	double ucb = -1;
	Node* parent = nullptr;
	std::vector<Node*> children;
	board state;
	action::place move;
	board::piece_type who;

	~Node(){};
};

/**
 * random player for both side
 * put a legal piece randomly
 */
class player : public random_agent {
public:
	player(const std::string& args = "") : random_agent("name=random role=unknown " + args),
		space(board::size_x * board::size_y), who(board::empty) {
		if (name().find_first_of("[]():; ") != std::string::npos)
			throw std::invalid_argument("invalid name: " + name());
		if (role() == "black") who = board::black;
		if (role() == "white") who = board::white;
		if (who == board::empty)
			throw std::invalid_argument("invalid role: " + role());
		if (meta.find("search") != meta.end()) search_method = (std::string)meta["search"];
		if (meta.find("timeout") != meta.end()) timeout = (clock_t)meta["timeout"];
		if (meta.find("simulation") != meta.end()) simulation_count = (int)meta["simulation"];
		for (size_t i = 0; i < space.size(); i++)
			space[i] = action::place(i, who);
	}

	virtual action take_action(const board& state) {
		if (search_method == "random") {
			std::shuffle(space.begin(), space.end(), engine);
			for (const action::place& move : space) {
				board after = state;
				if (move.apply(after) == board::legal)
					return move;
			}
			return action();
		}
		else if (search_method == "MCTS"){
			clock_t start_time = clock();
			clock_t end_time, total_time = 0;
			
			int total_count = 0;
			board::piece_type winner;
			Node* root = new Node;
			root->state = state;			
			root->who = (who == board::white ? board::black : board::white);
			Expansion(root);

			// default time limit = 1s //
			if (simulation_count == 0) { 
				while(total_time < timeout) {								
					Node* best_node = Selection(root);
					Expansion(best_node);
					winner = Simulation(best_node);
					total_count++;
					BackPropagation(root, best_node, winner, total_count);
					end_time = clock();
					total_time = (double)(end_time-start_time);
				}	
			}
			else {
				for (int count = 0; count < simulation_count; count++) {
					Node* best_node = Selection(root);
					Expansion(best_node);
					winner = Simulation(best_node);
					total_count++;
					BackPropagation(root, best_node, winner, total_count);
					
					//tree_policy(root, winner, total_visit_count);
				}
			}
			action best_action = BestAction(root);
			DeleteTree(root);
			//free(root);
			return best_action;

		}
	}

	Node* Selection(Node* node) {
		while(!node->children.empty()) {
			double max_ucb = -1;
			int select = 0;
			for(int i = 0; i < node->children.size(); i++) {
				if(node->children[i]->ucb > max_ucb) {
					max_ucb = node->children[i]->ucb;
					select = i;
				}
			}
			node = node->children[select];
		}
		return node;
	}

	void Expansion(Node* parent) {	
		if (parent->who == board::black) {
			for (int i = 0; i < space.size(); i++) {
				board after = parent->state;
				if (space[i].apply(parent->state) == board::legal) {
					Node* child = new Node;
					child->state = after;
					child->move = space[i];
					child->who = board::white;
					child->parent = parent;
					parent->children.emplace_back(child);
				}
			}
		}
		else if (parent->who == board::white) {
			for (int i = 0; i < space.size(); i++) {
				board after = parent->state;
				if (space[i].apply(after) == board::legal) {
					Node* child = new Node;
					child->state = after;
					child->move = space[i];
					child->who = board::black;
					child->parent = parent;
					parent->children.emplace_back(child);
				}
			}
		}	
	}

	board::piece_type Simulation(Node* root) {
		bool continue_flag = true;
		board state = root->state;
		board::piece_type who = root->who;
		
		while(continue_flag) {
			/* there are no legal move -> stop */
			continue_flag = false;
			
			/*rival's round*/
			if (who == board::white) who = board::black;
			else if (who == board::black) who = board::black;
				
			/* place randomly , apply the first legal move*/
			std::shuffle(space.begin(), space.end(), engine);
			for (int i = 0; i < space.size(); i++) {
				board after = state;
				if (space[i].apply(after) == board::legal) {
					space[i].apply(state);
					continue_flag = true;
					break;
				}
			}
		}
		/*I have no legal move, rival win*/
		return (who == board::white ? board::black : board::white);
	}

	double get_UCB_value(Node* node, int total_count) {
		double win_rate = (float) node->win / (float) node->total;
		double exploitation = sqrt(log(total_count) / (float) node->total);
		return win_rate + constant * exploitation;
	}

	void BackPropagation(Node* root, Node* node, board::piece_type winner, int total_count) {
		/* e.g.
		// root state : last_action = white 
		// -> root who = black 
		*/
		bool win = true;
		if(winner == root->who)
			win = false;
		while(node != nullptr) {
			node->total++;
			if(win == true) node->win++;
			node->ucb = get_UCB_value(node, total_count);
			node = node->parent;
		}
	}

	action BestAction(Node* node) {
		action best_action = action();
		int max_visit_count = 0;
		for(int i = 0; i < node->children.size(); i++) {		
			if(node->children[i]->total > max_visit_count) {
				max_visit_count = node->children[i]->total;
				best_action = node->children[i]->move;
			}
		}
		return best_action;
	}

	void DeleteTree(Node* node) {
		if(!node->children.empty()) {
			for(int i = 0; i < node->children.size(); i++) {
				DeleteTree(node->children[i]);
				//if(node->children[i] != nullptr)
				//	free(node->children[i]);
			}
			node->children.clear();
		}
	}

private:
	std::vector<action::place> space;
	board::piece_type who;
	std::string search_method;
	clock_t timeout = 1000;
	int simulation_count = 0;
	double constant = 0.1;
};