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
			/*
			 * mcts player with time management 
			 */
            clock_t start_time = clock();
            clock_t end_time;

            int total_count = 0;
			/* find out what step it is */
            int step = 73;
            for(int i = 0; i < 9; i++){
                for(int j = 0; j < 9; j++){
                    if(state[i][j] == board::empty) step--;
                }
            }
			/* It's opponent's turn */
            Node* root = new Node;
            root->state = state;
            root->who = (who == board::white ? board::black : board::white);
            Expansion(root);

			bool win;
			board::piece_type winner;
			Node* selected_node;
			/* If total time exceeds the limit time in this step, stop doing MCTS. */
            do {
				total_count++; /* record visit count in total */
                selected_node = Selection(root);
                Expansion(selected_node);
                winner = Simulation(selected_node);
                win = (root->who != winner); 
                BackPropogation(root, selected_node, win, total_count);
                end_time = clock();
            } while((double)(end_time - start_time)/CLOCKS_PER_SEC < time_management[step/2]);
            /* according to nodes’ # of totals, return the best action. */
			action best_action = get_best_action(root);
            delete_tree(root);
            delete(root);
            return best_action;
        }
        else {
			/**
			 * random player for both side
			 * put a legal piece randomly
			 */
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
		/* return leaf node */
		while(!node->children.empty()){
			double max_ucb = -1;
			int max_ucb_ind= 0; 
			/* select the child which has max ucb score */
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
			/* expand all the valid moves in white space */
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
			/* expand all the valid moves in black space */
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
            who = (who == board::white ? board::black : board::white); /* whose turn */
			if (who == board::black){
				/* black randomly gets one of valid moves */
				std::shuffle(black_space.begin(), black_space.end(), engine);
				for (size_t i = 0; i < black_space.size(); i++) {
					board after = state;
					if (black_space[i].apply(after) == board::legal){
						black_space[i].apply(state);
						flag = true;
						break;
					}
				}
			}
			else if (who == board::white){
				/* white randomly gets one of valid moves */
				std::shuffle(white_space.begin(), white_space.end(), engine);
				for (size_t i = 0; i < white_space.size(); i++) {
					board after = state;
					if (white_space[i].apply(after) == board::legal){
						white_space[i].apply(state);
						flag = true;
						break;
					}
				}
			}
		}
		/* no legal action, so the rival wins */
        who = (who == board::white ? board::black : board::white);
		return who; /* return the winner*/
	}

    void BackPropogation(Node* root, Node* node, bool win, int total_count){
		while(node != root){
			/* update the node’s # of wins and # of totals */
			node->total++;
			if (win) node->win++;
			/* update the node's ucb */
			node->ucb = ((double)node->win/(double)node->total) + constant * sqrt(log((double)total_count)/(double)node->total);
			node = node->parent;
		}
	}

    action get_best_action(Node* node){
		action best_action = action();
		int max_count = -1;
		/* select the best action based on the visit count of child nodes. */
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
                delete(node->children[i]);
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
    double constant = 0.5;
    double time_management[36] = {0.5, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 0.8, 
                                  1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 
                                  2.0, 2.0, 2.0, 2.0, 1.5, 1.5, 1.5, 1.5, 
                                  1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 
                                  0.5, 0.5, 0.5, 0.5};
};