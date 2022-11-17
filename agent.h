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
#include <iostream>
#include <cmath>
#include <map>
#include <type_traits>
#include <algorithm>
#include <fstream>
#include <time.h>
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

/**
 * random player for both side
 * put a legal piece randomly
 */
class Node {
	public:
        int total = 0; 
		int win = 0;
		double ucb = 10e9;
		Node* parent = nullptr;
		action::place move;
		std::vector<Node*> children;
        board state;
		board::piece_type who;
		~Node(){}
};

class player : public random_agent {
public:
	player(const std::string& args = "") : random_agent("name=random role=unknown " + args),
		space(board::size_x * board::size_y), white_space(board::size_x * board::size_y),
        black_space(board::size_x * board::size_y), who(board::empty) {
		if (name().find_first_of("[]():; ") != std::string::npos)
			throw std::invalid_argument("invalid name: " + name());
        if (meta.find("search") != meta.end()) search = (std::string)meta["search"];
		if (role() == "black") who = board::black;
		if (role() == "white") who = board::white;
		if (who == board::empty)
			throw std::invalid_argument("invalid role: " + role());
		for (size_t i = 0; i < space.size(); i++)
			space[i] = action::place(i, who);
        for (size_t i = 0; i < white_space.size(); i++)
			white_space[i] = action::place(i, board::white);
		for (size_t i = 0; i < black_space.size(); i++)
			black_space[i] = action::place(i, board::black);
	}

	virtual action take_action(const board& state) {
        if (search == "mcts") {
            clock_t start_time = clock();
            clock_t end_time;
            int total_count = 0;
            int step = 73;
            for(int i = 0; i < 9; i++){
                for(int j = 0; j < 9; j++){
                    if(state[i][j] == board::empty) step--;
                }
            }
            Node* root = new Node;
            root->state = state;
            root->who = (who == board::white ? board::black : board::white);
            Expansion(root);
            while(1) {
                Node* selected_node = Selection(root);
                Expansion(selected_node);
                board::piece_type winner = Simulation(selected_node);
                bool win = (root->who != winner);
                total_count++;
                BackPropogation(selected_node, win, total_count);
                end_time = clock();
                double total_time = (double)(end_time - start_time)/CLOCKS_PER_SEC;
                if (total_time >= time_management[step/2]) break;
            }
            action best_action = get_best_action(root);
            delete_tree(root);
            free(root);
            return best_action;
        }
        else {
            std::shuffle(space.begin(), space.end(), engine);
            for (const action::place& move : space) {
                board after = state;
                if (move.apply(after) == board::legal)
                    return move;
            }
        }
		return action();
	}

    Node* Selection(Node* node){
		while(!node->children.empty()){
			double max_ucb = -1;
			int max_ucb_ind= 0; 
			for(size_t i=0; i < node->children.size(); i++){
				if (node->children[i]->ucb > max_ucb){
					max_ucb = node->children[i]->ucb;
					max_ucb_ind = i;
				}
			}
			node = node->children[max_ucb_ind];
		}
		return node;
	}

    void Expansion(Node* parent_node){
		if (parent_node->who == board::black){
			for (size_t i = 0; i < white_space.size(); i++){
				board after = parent_node->state;
				if (white_space[i].apply(after) == board::legal){
					Node* child_node = new Node;
					child_node->who = board::white;
					child_node->state = after;
                    child_node->move = white_space[i];
					child_node->parent = parent_node;
					parent_node->children.emplace_back(child_node);
				}
			}
		}
		else if(parent_node->who == board::white){
			for (size_t i = 0; i < black_space.size(); i++){
				board after = parent_node->state;
				if (black_space[i].apply(after) == board::legal){
					Node* child_node = new Node;
					child_node->who = board::black;
					child_node->state = after;
                    child_node->move = black_space[i];
					child_node->parent = parent_node;
					parent_node->children.emplace_back(child_node);
				}
			}
		}
	}

    board::piece_type Simulation(Node* node){
		bool flag = true;
		board state = node->state;
		board::piece_type who = node->who;
		while(flag){
			flag = false;
            who = (who == board::white ? board::black : board::white);
			if (who == board::black){
				std::shuffle(black_space.begin(), black_space.end(), engine);
				for (const action::place& move : black_space) {
					board after = state;
					if (move.apply(after) == board::legal){
						move.apply(state);
						flag = true;
						break;
					}
				}
			}
			else if (who == board::white){
				std::shuffle(white_space.begin(), white_space.end(), engine);
				for (const action::place& move : white_space) {
					board after = state;
					if (move.apply(after) == board::legal){
						move.apply(state);
						flag = true;
						break;
					}
				}
			}
		}
        who = (who == board::white ? board::black : board::white);
		return who;
	}

    double get_UCB_value(Node* node, int total_visit_count){
		return ((double)node->win/node->total) + constant * sqrt(log((double)total_visit_count)/node->total);
	}

    void BackPropogation(Node* node, bool win, int total_count){
		while(node != nullptr){
			node->total++;
			if (win) node->win++;
			node->ucb = get_UCB_value(node, total_count);
			node = node->parent;
		}
	}

    action get_best_action(Node* node){
		action best_action = action();
		int max_count = -1;
		for(size_t i = 0; i < node->children.size(); i++){
			if (node->children[i]->total > max_count){
				max_count = node->children[i]->total;
				best_action = node->children[i]->move;
			}
		}
		return best_action;
	}

    void delete_tree(Node* node){
		if(!node->children.empty()){
			for(size_t i = 0; i < node->children.size(); i++) {
                delete_tree(node->children[i]);
                free(node->children[i]);
            }
			node->children.clear();
		}
	}

private:
    std::string search;
	std::vector<action::place> space;
    std::vector<action::place> white_space;
    std::vector<action::place> black_space;
	board::piece_type who;
    double constant = 0.1;
    double time_management[36] = {0.3, 0.3, 0.3, 0.3, 0.6, 0.6, 0.6, 0.6, 
                                  0.9, 0.9, 0.9, 0.9, 1.2, 1.2, 1.2,1.2, 
                                  1.5, 1.5, 1.5, 1.5, 1.2, 1.2, 1.2, 1.2, 
                                  0.9, 0.9, 0.9, 0.9, 0.6, 0.6, 0.6, 0.6, 
                                  0.3, 0.3, 0.3, 0.3};
};