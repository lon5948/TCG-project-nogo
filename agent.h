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
#include <omp.h>
#include <thread>

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

class node{
	public:
		board state;
		board::piece_type who;
		int win = 0;
		int visit = 0;
		//double UCB_RAVE_value = 0x7fffffff;
		action::place move;
		node* parent = nullptr;
		std::vector<node*> children;
		~node(){};
};

class MCTS_player : public random_agent {
public:
	std::vector<action::place> space, white_space, black_space;
	MCTS_player(const std::string& args = "") : random_agent("name=random role=unknown " + args),
		space(board::size_x * board::size_y),white_space(board::size_x * board::size_y),
		black_space(board::size_x * board::size_y), who(board::empty) {
		if (meta.find("search") != meta.end()) search = (std::string)meta["search"];
		if (meta.find("simulation") != meta.end()) simulation_count = (int)meta["simulation"];
		if (meta.find("thread") != meta.end()) thread_num = (int)meta["thread"];
		if (role() == "black") who = board::black;
		if (role() == "white") who = board::white;
		for (size_t i = 0; i < space.size(); i++)
			space[i] = action::place(i, who);
		for (size_t i = 0; i < white_space.size(); ++i)
			white_space[i] = action::place(i, board::white);
		for (size_t i = 0; i < black_space.size(); ++i)
			black_space[i] = action::place(i, board::black);
	}

	virtual action take_action(const board& state) {
		if (search == "p-mcts"){
			omp_set_num_threads(thread_num);
			std::vector<node*> roots(thread_num);

			#pragma omp parallel for
			for(int i = 0; i < thread_num; i++) {
				roots[i] = new node;
				roots[i]->state = state;
				roots[i]->who = (who == board::white ? board::black : board::white);
				int total_node = 0;
				Expansion(roots[i], total_node);
				board::piece_type winner;
				run_MCTS(roots[i], winner, total_node);	
			}		

			for (int idx = 1; idx < thread_num; idx++) {
				for(size_t i = 0; i < roots[0]->children.size() ; i++) {
					roots[0]->children[i]->visit += roots[idx]->children[i]->visit;
				}
			}

			action best_action = get_action(roots[0]);
			#pragma omp parallel for
			for(int i = 0; i < thread_num; i++) {
				delete_tree(roots[i]);
				free(roots[i]);
			}
			return best_action;
		}
		else {
			std::shuffle(space.begin(), space.end(), engine);
			for (const action::place& move : space) {
				board after = state;
				if (move.apply(after) == board::legal)
					return move;
			}
			return action();
		}
	}

	node* Selection(node* n) {			
		node* cur = n;
		while(!cur->children.empty()) {
			double max_value = 0;
			int select_idx = 0;
			for(size_t i = 0; i < cur->children.size(); ++i) {
				double ucb = get_ucb_value(cur->children[i]);
				if(max_value < ucb) {
					max_value = ucb;
					select_idx = i;
				}
			}
			cur = cur->children[select_idx];
		}
		return cur;
	}

	double get_ucb_value(node* cur) {					
		if(cur->visit == 0 || rave_map[cur->move].first == 0) return 1e8;
		double constant = std::sqrt(2);
		double beta = std::sqrt((double) simulation_count/(double)(3 * count + simulation_count));
		double win_rate = (double) cur->win / (double) cur->visit;
		double rave_win_rate = (double) rave_map[cur->move].second / (double) rave_map[cur->move].first;
		double exploitation = (1 - beta) * win_rate + beta * rave_win_rate;
		double exploration = sqrt(log((double)cur->parent->visit)/cur->visit);
		return exploitation + constant * exploration;
	}
	
	void Expansion(node* parent_node, int& total_node) {
		action::place child_move;
		if (parent_node->who == board::black) {
			for(const action::place& child_move : white_space) {
				board after = parent_node->state;
				if (child_move.apply(after) == board::legal) {
					node* child_node = new node;
					child_node->state = after;
					child_node->parent = parent_node;
					child_node->move = child_move;
					child_node->who = board::white;
					parent_node->children.emplace_back(child_node);
					if (rave_map.find(child_node->move) == rave_map.end()) 
						rave_map.insert(std::make_pair(child_node->move, std::make_pair(0, 0)));
				}
			}
		}
		else if (parent_node->who == board::white) {
			for(const action::place& child_move : black_space) {
				board after = parent_node->state;
				if (child_move.apply(after) == board::legal) {
					node* child_node = new node;
					child_node->state = after;
					child_node->parent = parent_node;
					child_node->move = child_move;
					child_node->who = board::black;
					parent_node->children.emplace_back(child_node);
					if (rave_map.find(child_node->move) == rave_map.end()) 
						rave_map.insert(std::make_pair(child_node->move, std::make_pair(0, 0)));
				}
			}
		}	
		total_node += parent_node->children.size();
	}				
	
	board::piece_type Simulation(node* root) {
		bool terminal = false;
		board state = root->state;
		board::piece_type nodewho = root->who;
		while(!terminal) {
			terminal = true;
			nodewho = (nodewho == board::white ? board::black : board::white);
			if (nodewho == board::black) {
				std::shuffle(black_space.begin(), black_space.end(), engine);
				for (size_t i = 0; i < black_space.size(); i++) {
					board after = state;
					if (black_space[i].apply(after) == board::legal) {
						black_space[i].apply(state);
						terminal = false;
						break;
					}
				}
			}
			else if (nodewho == board::white){
				std::shuffle(white_space.begin(), white_space.end(), engine);
				for (size_t i = 0; i < white_space.size(); i++) {
					board after = state;
					if (white_space[i].apply(after) == board::legal){
						white_space[i].apply(state);
						terminal = false;
						break;
					}
				}
			}
		}
		return (nodewho == board::white ? board::black : board::white);
	}
	
	void BackPropagation(node* root, node* cur, board::piece_type winner) {
		while(cur != root) {
			cur->visit += 1;
			rave_map[cur->move].first += 1;
			if(winner != root->who){
				cur->win += 1;
				rave_map[cur->move].second += 1;
			}
			cur = cur->parent;
		}
		root->visit += 1;
		rave_map[root->move].first += 1;
		if(winner != root->who) {
			root->win += 1;
			rave_map[root->move].first += 1;
		}
	}
	
	void run_MCTS(node* root, board::piece_type winner, int total_node){
		for (int i = 0; i < simulation_count; i++) {
			node* best_node = Selection(root);
			Expansion(best_node, total_node);	
			if(best_node->children.size() != 0){
				std::shuffle(best_node->children.begin(), best_node->children.end(), engine);
				winner = Simulation(best_node->children[0]);
				BackPropagation(root, best_node->children[0], winner);
			}
			else{
				winner = Simulation(best_node);
				BackPropagation(root, best_node, winner);
			}
			count += 1;
		}
	}
	
	action get_action(node* root) {					
		int child_idx = -1;
		int max_visit = 0;
		for(size_t i = 0; i < root->children.size(); ++i) {
			if(root->children[i]->visit > max_visit) {
				max_visit = root->children[i]->visit;
				child_idx = i;
			}
		}
		if(child_idx != -1) return root->children[child_idx]->move;
		return action();
	}
	
	void delete_tree(node* node) {
		if(node->children.empty() == false) {
			for(size_t i = 0; i < node->children.size(); ++i) {
				delete_tree(node->children[i]);
				if(node->children[i] != NULL)
					free(node->children[i]);
			}
			node->children.clear();
		}
		return;
	}

private:
	std::string search;
	int simulation_count = 0;
	int count = 0;
	int thread_num = 4;
	board::piece_type who;
	std::map<action::place, std::pair<int, int> > rave_map;
	double time_management[36] = {	5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
									6.0, 5.0, 5.0, 5.0, 5.0, 5.0,
									9.0, 9.0, 9.0, 9.0, 9.0, 9.0,
									10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
									10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
									10.0, 10.0, 10.0, 10.0, 10.0, 1.0
								};
};