# Copyright (c) 2019 Shikha Mittal. All rights reserved.
# Implementation of iFALCON (Initial Fuzzy Adaptive Logical Cognitive Neural) architecture.
# Developed independently as a functional replication of cognitive planning research.

#!/usr/bin/env python
# coding: utf-8

# In[306]:


import queue


# In[307]:


class InputField:                                                           #for F1 input/output field                                         
    def __init__(self,itype):                                                     
        self.I = []                                                         #input vector
        self.X = []                                                         #activity vector
        self.alpha  = 0                                                     #choice parameter
        self.beta = 0                                                       #learning parameter 
        self.gamma = 0                                                      #contribution parameter
        self.rho = 0                                                        #vigilance parameter
        self.itype = itype                                                  # input type to be used for 0 as belief, 
                                                                            #1 as critic, 2 as desire , 3 as action 
            
    def set_input_vector(self, input_vector):                               #setter method for input
        self.I = input_vector                                               #sets input 
        self.set_activity_vector(input_vector)                              #also updates the activity vector
        
    def set_activity_vector(self, activity_vector):                         #setter method for activity_vector
        self.X = activity_vector
                
    def set_alpha(self,alpha):                                              #setter method for alpha
        self.alpha = alpha
        
    def set_beta(self,beta):                                                #setter method for beta
        self.beta = beta
        
    def set_gamma(self,gamma):                                              #setter method for gamma
        self.gamma = gamma
        
    def set_rho(self,rho):                                                  #setter method for rho
        self.rho = rho
        
    def set_parameters(self,alpha,beta,gamma,rho):                          #to set all parameters at once                            
        self.alpha  = alpha
        self.beta = beta
        self.gamma = gamma
        self.rho = rho


# In[308]:


class F2CategoryField:
    
    node_cnt = 0                                                            #to be used to keep count of number of nodes
    node_lst =[]                                                            #list of nodes in the category field 
    
    #update this to clone input to weight vector
    #whenever a new node is created, it is initialized 
    #to input vectors
    def __init__(self):
        self.weight_belief = []                                             #weight vector for belief field
        self.weight_critic = []                                             #weight vector for critic field
        self.weight_desire = []                                             #weight vector for desire field
        self.choice_fn = 0                                                  #choice function
        self.y = 0                                                          #to be used to indicate which node is active                                                 
        F2CategoryField.node_cnt = F2CategoryField.node_cnt+1               #keeps track of allocated nodes in category field
        self.id = F2CategoryField.node_cnt 
        self.action_seq = []                                                #will contain nodes of F3 in sequence
        
    def set_weight_belief(self, new_weight_belief):                         #weight with the belief field
        self.weight_belief = new_weight_belief
        
    def set_weight_critic(self, new_weight_critic):                         #weight with the critic field
        self.weight_critic = new_weight_critic
        
    def set_weight_desire(self, new_weight_desire):                         #weight with the desire field
        self.weight_desire = new_weight_desire
        
    def init_weights(self,w_belief, w_critic,w_desire):                     #to set all the weights if needed
        self.set_weight_belief(w_belief)
        self.set_weight_critic(w_critic)
        self.set_weight_desire(w_desire)
    
    def set_y(self, new_y):                                                 #to be used to indicate activation of node
        self.y = new_y                                                      #if 1, node is active/fired , 0 otherwise
        
    def set_choice_function(self,new_choice_fn):                            #to be used to calculate node choice function 
        self.choice_fn = new_choice_fn                                      #node with highest choice function will be 
                                                                            #selected first for resonance check  
    
    #class static method
    #maintains list of all available F2 nodes
    def add_to_available_nodes(obj_F2):  
        F2CategoryField.node_lst.append(obj_F2)                             #add to list of available nodes 
        
    #choice function will be calculated every 
    #time input/output field is presented
    #this is only calculated after input presentation to ip/op field
    #to be used to select node with max Tj (choice_fn )
    #node with max_choice_fn is used to verify resonance
    def calc_total_choice_fn(self, obj_belief, obj_critic, obj_desire):     #calculates choice fn for a node j in F2
        total_choice_fn = 0
        total_choice_fn = total_choice_fn +                                                        IFalcon.calc_choice_fn_field(                                              obj_belief,self.weight_belief)
        total_choice_fn = total_choice_fn +                                                       IFalcon.calc_choice_fn_field(                                               obj_critic,self.weight_critic)         
        total_choice_fn = total_choice_fn +                                                        IFalcon.calc_choice_fn_field(                                              obj_desire,self.weight_desire)
        self.set_choice_function(total_choice_fn)
    
    #vigilance is checked for each node, for top down verification
    def isVigilanceConstraintSatisfied(self,obj_belief, obj_critic , obj_desire):
        return IFalcon.calc_match_fn_field(obj_belief.X, self.weight_belief)                                                         >= obj_belief.rho and                               IFalcon.calc_match_fn_field(obj_critic.X, self.weight_critic)                                                         >= obj_critic.rho and                               IFalcon.calc_match_fn_field(obj_desire.X, self.weight_desire)                                                         >= obj_desire.rho
    
    #learn template
    def updateWeights(self,obj_belief, obj_critic, obj_desire):
        self.weight_belief = IFalcon.updateWeightsField(obj_belief,self.weight_belief)
        self.weight_critic = IFalcon.updateWeightsField(obj_critic,self.weight_critic)
        self.weight_desire = IFalcon.updateWeightsField(obj_desire,self.weight_desire)
        
        
    #will take current activity vectors as input 
    #and list of all available F2 Nodes
    import queue
    def createPriorityQueue(node_list,obj_belief,obj_critic,obj_desire):
        pq = queue.PriorityQueue()
        
        for node in node_list:                                                  #iterate through every node in the list
            
            node.calc_total_choice_fn(obj_belief,obj_critic,obj_desire)         #calculate choice fn of the node
            
            j = ((-node.choice_fn,node.id),node)                                #Add node to priority queue
                                                                                #adding -ve choice fn will enable to have 
                                                                                #first node with max choice fn
            pq.put(j)
        return pq                                                               #return priority queue


# In[309]:


class F3CategoryField:
    
    node_cnt = 0                                                               #to be used to keep count of number of nodes
    node_lst =[]                                                               #list of nodes in the category field 
    
    #update this to clone input to weight vector
    #whenever a new node is created, it is initialized 
    #to input vectors
    def __init__(self):
        self.weight_action = []                                                #weight vector for action field
        self.weight_desire = []                                                #weight vector for desire field
        self.choice_fn = 0                                                     #choice function
        self.y = 0                                                             #to be used to indicate which node is active                                                 
        F3CategoryField.node_cnt = F3CategoryField.node_cnt+1                  #keeps track of allocated nodes in category field
        self.id = F3CategoryField.node_cnt 
        
        
    def set_weight_action(self, new_weight_action):                            #weight with the action field
        self.weight_action = new_weight_action
        
    def set_weight_desire(self, new_weight_desire):                            #weight with the desire field
        self.weight_desire = new_weight_desire
        
    def init_weights(self,w_action,w_desire):                                  #to set all the weights if needed
        self.set_weight_action(w_action)
        self.set_weight_desire(w_desire)
    
    def set_y(self, new_y):                                                    #to be used to indicate activation of node
        self.y = new_y                                                         #if 1, node is active/fired , 0 otherwise
        
    def set_choice_function(self,new_choice_fn):                               #to be used to calculate node choice function 
        self.choice_fn = new_choice_fn                                         #node with highest choice function will be 
                                                                               #selected first for resonance check 
            
    #class static method
    #maintains list of all available F2 nodes
    def add_to_available_nodes(obj_F3):  
        F3CategoryField.node_lst.append(obj_F3)                                #add to list of available nodes 
    
    #choice function will be calculated every 
    #time input/output field is presented
    #this is only calculated after input presentation to ip/op field
    #to be used to select node with max Tj (choice_fn )
    #node with max_choice_fn is used to verify resonance
    def calc_total_choice_fn(self, obj_action, obj_desire):                    #calculates choice fn for a node j in F2
        total_choice_fn = 0
        total_choice_fn = total_choice_fn +                                                           IFalcon.calc_choice_fn_field(                                                 obj_action,self.weight_action)
        total_choice_fn = total_choice_fn +                                                           IFalcon.calc_choice_fn_field(                                                 obj_desire,self.weight_desire)            
        self.set_choice_function(total_choice_fn)
        
    def isVigilanceConstraintSatisfied(self,obj_action, obj_desire):
        return IFalcon.calc_match_fn_field(obj_action.X, self.weight_action)                                                         >= obj_action.rho and                               IFalcon.calc_match_fn_field(obj_desire.X, self.weight_desire)                                                         >= obj_desire.rho
    
    #learn template
    def updateWeights(self,obj_action, obj_desire):
        self.weight_action = IFalcon.updateWeightsField(obj_action,self.weight_action)
        self.weight_desire = IFalcon.updateWeightsField(obj_desire,self.weight_desire)
        
    #will take current activity vectors as input 
    #and list of all available F3 Nodes
    import queue
    def createPriorityQueue(node_list,obj_action,obj_desire):
        pq = queue.PriorityQueue()
        
        for node in node_list:                                                 #iterate through every node in the list
            
            node.calc_total_choice_fn(obj_action,obj_desire)                   #calculate choice fn of the node
            
            j = ((-node.choice_fn,node.id),node)                               #Add node to priority queue
                                                                               #adding -ve choice fn will enable to have 
                                                                               #first node with max choice fn
            pq.put(j)
        return pq                                                              #return priority queue
    


# In[310]:


class IFalcon:

    
    def fuzzyAnd(x1,x2):                                                        #for Choice function x1 is x_cl and x2 is w_cl
        result = []
        for i,j in zip(x1,x2):
            result.append( min(i,j))  
        return result
    
    def norm(x):                                                                #Calculates norm
        sum = 0
        for i in x:
            sum = sum + i
        return sum  

    # called by calc_total_choice_fn for every input field
    def calc_choice_fn_field(obj_input,weights):                                #calculates choice fn for a input field
        
        numerator = IFalcon.norm(                                                    IFalcon.fuzzyAnd(obj_input.X,weights))         
        
        denominator = obj_input.alpha + IFalcon.norm(weights)
        
        temp_choice_fn =  ((numerator/denominator) * obj_input.gamma)
        
        return temp_choice_fn
    
    #returns value of the match function for the given
    #weight and activity vectors
    #will be called to check for resonance
    #this is checked for each input channel
    def calc_match_fn_field(x, w):                                              #calculates match function
        isNull = True
        for i in x:
            if(i != 0):                                                         #if even a single 1 in x then it cannot be zero
                isNull = False
        
        if (isNull): return 1
        
        denominator = IFalcon.norm(x)
        if ( denominator == 0 ): return 1
        return IFalcon.norm(IFalcon.fuzzyAnd(x, w))/denominator
    
    def updateWeightsField(obj_input, weight):
        first_part = [ i * (1 - obj_input.beta) for i in weight]
        second_part = [ i * obj_input.beta                                             for i in IFalcon.fuzzyAnd(obj_input.X,weight)]
        new_weight =[ first_part[i] + second_part[i] for i in range(len(first_part))]
        return new_weight
    
    def addPlanNode(obj_belief,obj_critic, obj_desire):
        
        pNode = F2CategoryField()                                               #create plan node
        F2CategoryField.add_to_available_nodes(pNode)                           #add to list of available nodes
        pNode.init_weights(obj_belief.X,obj_critic.X,obj_desire.X)              #update weight vectors
        return pNode                                                            #creates plan Node and returns node
    
    import queue
    def planNodeEncoding(obj_belief,obj_critic,obj_desire,obj_action):
        
        pq = queue.PriorityQueue()
        
        #First we need to find if there is existing plan
        #and if that plan is close to what we are trying to encode
        #if it is close to what we are trying to encode
        #then just update the existing one
        if F2CategoryField.node_cnt == 0:                                       #No node in F2, no plan exists

            pNode = IFalcon.addPlanNode(obj_belief,obj_critic,obj_desire)       #Add a plan node
        
        else:                                                                   #Plan Nodes in F2 exists
                                                                                #check for resonance, if existing plan is close
                                                                                #to one we are trying to encode
            pq = F2CategoryField.createPriorityQueue(                                F2CategoryField.node_lst,obj_belief,obj_critic,obj_desire)
            nodeFound = False                                                   #to be used to indicate existing plan found or not
            while not pq.empty():                                               #checking node with highest choice fn
                head = pq.get()
                
                if (head[1].isVigilanceConstraintSatisfied(                                 obj_belief,obj_critic,obj_desire) == True):
                    nodeFound = True                                               #Head will contain the pointer for existing plan node
                    break;                                                         #no need to check further nodes
                
            if nodeFound == True:

                pNode = head[1]                                                    #assign the existing node to plan node
                pNode.updateWeights(obj_belief, obj_critic, obj_desire)
            else:

                pNode = IFalcon.addPlanNode(obj_belief,obj_critic,obj_desire)      #checked all existing plan nodes but none 
                                                                                   #matches the criteria for vigilance
        return pNode           
        
    #Now we have plan Node, it is either new Node or an 
    #existing Node
    #We need to check for action sequence now
    #in any case we will update the action sequence to one 
    #we want to encode
    #for updation of action sequence, we can clear action_seq
    #already attached with the node if any and encode new one here
    def addActionNode(obj_action,obj_desire):
        
        aNode = F3CategoryField()                                                #create plan node
        F3CategoryField.add_to_available_nodes(aNode)                            #add to list of available nodes
        aNode.init_weights(obj_action.X,obj_desire.X)                            #update weight vectors
        return aNode 
    

                    
    #we need to find F3/action node
    #check if action exsits,
    #if No, then encode new one
    #if yes, then modify it
    import queue
    def actionNodeEncoding(obj_action,obj_desire):
    
        pq = queue.PriorityQueue()
        
        #First we need to find if there is existing action
        #and if that action is close to what we are trying to encode
        #if it is close to what we are trying to encode
        #then just update the existing one
        if F3CategoryField.node_cnt == 0:                                     #No node in F3, no action exists
           
            aNode = IFalcon.addActionNode(obj_action,obj_desire)              #Add an action node
        else:                                                                 #Action Nodes in F3 exists
                                                                              #check for resonance, if existing action is close
                                                                              #to one we are trying to encode
            pq = F3CategoryField.createPriorityQueue(                                F3CategoryField.node_lst,obj_action,obj_desire)
            nodeFound = False                                                 #to be used to indicate existing action found or not
            while not pq.empty():                                             #checking node with highest choice fn
                head = pq.get()
                if (head[1].isVigilanceConstraintSatisfied(                                 obj_action,obj_desire) == True):
                    nodeFound = True                                          #Head will contain the pointer for existing action node
                    break;                                                    #no need to check further nodes
                
            if nodeFound == True:
                aNode = head[1]                                               #assign the existing node to action node
                aNode.updateWeights(obj_action, obj_desire) 
            else:
                aNode = IFalcon.addActionNode(obj_action,obj_desire)          #checked all existing action nodes but none 
                                                                              #matches the criteria for vigilance
        return aNode
              
                  
    #for now works only for primitive plans
    #this can be called as many times
    #for main program when we want to feed
    #the plan
    def planLearning(belief_cl,critic_cl,desire_cl, action_cl):
        
        
        #as we have all the inputs now we 
        #need action and plan node
        pNode = IFalcon.planNodeEncoding(belief_cl,critic_cl,desire_cl,action_cl)
        aNode = IFalcon.actionNodeEncoding(action_cl, desire_cl)
        
        #add code to link plan to the action
        pNode.action_seq.clear()                                              #in case there is existing list
        pNode.action_seq.append(aNode)
       
        
    #************************************************************************#
    #*******************Plan Selection and Execution Function****************#
    #************************************************************************#
   
   
    #Try to find existing plan
    #if existing plan is not found then a new plan node
    #is recruited
    def planSelection(obj_belief,obj_critic,obj_desire):
        
        head=""
        
        #trying to find existing node
        pq = F2CategoryField.createPriorityQueue(                                F2CategoryField.node_lst,obj_belief,obj_critic,obj_desire)
        nodeFound = False                                                   #to be used to indicate existing plan found or not
        while not pq.empty():                                               #checking node with highest choice fn
            head = pq.get()
            if (head[1].isVigilanceConstraintSatisfied(                             obj_belief,obj_critic,obj_desire) == True):
                nodeFound = True                                            #Head will contain the pointer for existing plan node
                break;
               
        if nodeFound == True:
            pNode = head[1]                                                 #assign the existing node to plan node
        else:
            pNode = ""
        
        return pNode
    
    #need to consider what will happen in the case of denominator 
    #being zero
    def calc_match_fn_d(belief_cl,desire_cl):
        
        numerator = IFalcon.norm(IFalcon.fuzzyAnd(belief_cl.X, desire_cl.X))
        denominator = IFalcon.norm(desire_cl.X)
        
        return (numerator/denominator)
        

    def planSelectionAndExecution(belief_cl,critic_cl,desire_cl):
        
        delta = 0.1
        rho_s = desire_cl.rho
        
        while True:
            
            #perceive environment and update belief

            match_d = IFalcon.calc_match_fn_d(belief_cl,desire_cl)

            #critic evaluation
            if (match_d >= desire_cl.rho):
                #goal is satisfied
                break;
            else:
            
                epNode = IFalcon.planSelection(belief_cl,critic_cl,desire_cl)

                while (epNode == "" and desire_cl.rho >= 0.0):
                    desire_cl.set_rho(desire_cl.rho - delta)
                    epNode = IFalcon.planSelection(belief_cl,critic_cl,desire_cl)
                    
                desire_cl.set_rho(rho_s)
                belief_cl.set_input_vector(epNode.weight_desire)
                if (epNode == ""):
                    print('No plan found')
                    
                #existing plan with action found
                #print("EpNode action sequence",epNode.action_seq[0].weight_action)
                for i in epNode.action_seq:
                    print("\n")
                    print("Action weight",i.weight_action,"\n")
                    print("Desire weight",i.weight_desire,"\n")
                    #belief_cl.set_input_vector(i.weight_desire)

                

                

        


# In[311]:


class iFalconTest:
    
    def print_input_cl(input_cl):
        print("**********Channel Attributes************")
        print("Input vector:    ",input_cl.I)
        print("Activity vector: ",input_cl.X)
        print("Alpha: ",input_cl.alpha)
        print("Beta:  ",input_cl.beta)
        print("Gamma: ",input_cl.gamma)
        print("rho:   ",input_cl.rho)
        print("type:  ",input_cl.itype)
        print("***************************************")
        print('')
    
        
    def print_f2_node(jnode):
        print("*******Category Node Attributes*********")
        print("weight belief: ",jnode.weight_belief)
        print("weight critic: ",jnode.weight_critic)
        print("weight desire: ",jnode.weight_desire)
        print("Y : ",jnode.y)
        print("choice_fn: ",jnode.choice_fn)
        print('')
        
    #creating input channels    
    belief_cl = InputField(0)
    critic_cl = InputField(1)
    desire_cl = InputField(2)
    action_cl = InputField(3)
    
    #set configuration parameters
    #alpha,beta,gamma,rho
    belief_cl.set_parameters(1,1,1,1)
    desire_cl.set_parameters(1,1,1,1)
    critic_cl.set_parameters(1,1,1,1)
    action_cl.set_parameters(1,1,1,1)
    
    #plan1 - 1 
    belief_cl.set_input_vector([1,0,0,1,1,1])
    critic_cl.set_input_vector([1,1,1,1,1,1])
    desire_cl.set_input_vector([1,0,0,0,0,1])
    action_cl.set_input_vector([0,0,0,1,0,0])

    #learn first plan and action
    IFalcon.planLearning(belief_cl,critic_cl,desire_cl, action_cl)
    
    #plan5 - 2
    belief_cl.set_input_vector([1,1,0,1,1,0])   
    critic_cl.set_input_vector([1,1,1,1,1,1])
    desire_cl.set_input_vector([0,1,0,0,1,0])
    action_cl.set_input_vector([0,0,0,1,0,0])
    
    IFalcon.planLearning(belief_cl,critic_cl,desire_cl, action_cl)
 
    #plan2 - 3 
    belief_cl.set_input_vector([1,0,0,0,0,1])   
    critic_cl.set_input_vector([1,1,1,1,1,1])
    desire_cl.set_input_vector([0,0,0,0,0,0])
    action_cl.set_input_vector([0,0,0,0,0,1])
    
    
    IFalcon.planLearning(belief_cl,critic_cl,desire_cl, action_cl)
    
    #plan3 - 4
    belief_cl.set_input_vector([0,0,0,0,0,0])   
    critic_cl.set_input_vector([1,1,1,1,1,1])
    desire_cl.set_input_vector([0,0,0,1,1,0])
    action_cl.set_input_vector([0,0,1,0,0,0])
    
    
    IFalcon.planLearning(belief_cl,critic_cl,desire_cl, action_cl)
    
    #plan6 - 5
    belief_cl.set_input_vector([0,1,0,0,1,0])  
    critic_cl.set_input_vector([1,1,1,1,1,1])
    desire_cl.set_input_vector([0,0,0,0,0,0])
    action_cl.set_input_vector([0,1,0,0,0,0])
    
    IFalcon.planLearning(belief_cl,critic_cl,desire_cl, action_cl)
         
    #plan10 - 6
    belief_cl.set_input_vector([0,1,1,0,0,0])
    critic_cl.set_input_vector([1,1,1,1,1,1])
    desire_cl.set_input_vector([0,0,0,0,0,0])
    action_cl.set_input_vector([0,1,0,0,0,0])
    
    IFalcon.planLearning(belief_cl,critic_cl,desire_cl, action_cl)

    #plan12 - 7
    belief_cl.set_input_vector([1,0,0,1,0,0])
    critic_cl.set_input_vector([1,1,1,1,1,1])
    desire_cl.set_input_vector([0,0,0,0,0,0])
    action_cl.set_input_vector([0,0,0,1,0,0])
    
    IFalcon.planLearning(belief_cl,critic_cl,desire_cl, action_cl)
    
    #plan4 - 8
    belief_cl.set_input_vector([0,0,0,1,1,0])  
    critic_cl.set_input_vector([1,1,1,1,1,1])
    desire_cl.set_input_vector([0,1,1,1,1,0])
    action_cl.set_input_vector([1,0,0,0,0,0])
    
    IFalcon.planLearning(belief_cl,critic_cl,desire_cl, action_cl)

    #plan8 - 9
    belief_cl.set_input_vector([0,0,1,0,0,1])
    critic_cl.set_input_vector([1,1,1,1,1,1])
    desire_cl.set_input_vector([0,0,0,0,0,0])
    action_cl.set_input_vector([0,0,0,0,0,1])
    
    IFalcon.planLearning(belief_cl,critic_cl,desire_cl, action_cl)
     
    #plan11 - 10
    belief_cl.set_input_vector([1,0,1,1,0,1])
    critic_cl.set_input_vector([1,1,1,1,1,1])
    desire_cl.set_input_vector([1,0,0,1,0,0])
    action_cl.set_input_vector([0,0,0,0,0,1])
    
    IFalcon.planLearning(belief_cl,critic_cl,desire_cl, action_cl)
    
    #plan9 - 11
    belief_cl.set_input_vector([1,1,1,0,0,1])
    critic_cl.set_input_vector([1,1,1,1,1,1])
    desire_cl.set_input_vector([0,1,1,0,0,0])
    action_cl.set_input_vector([0,0,0,0,0,1])
    
    IFalcon.planLearning(belief_cl,critic_cl,desire_cl, action_cl)
   
    #plan7 - 12
    belief_cl.set_input_vector([0,1,1,0,1,1])
    critic_cl.set_input_vector([1,1,1,1,1,1])
    desire_cl.set_input_vector([0,0,1,0,0,1])
    action_cl.set_input_vector([0,1,0,0,0,0])
    
    IFalcon.planLearning(belief_cl,critic_cl,desire_cl, action_cl)
    
    print("Total Nodes in F2 category Field are: ",F2CategoryField.node_cnt)
    print("Total Nodes in F3 category Field are: ",F3CategoryField.node_cnt)
    


    #learnt plans
    for i in F2CategoryField.node_lst:
        print(i.id)
        print(i.weight_belief)
        print(i.weight_desire)
        print(i.action_seq[0].weight_action)
        
    print("\n")
    print("**************************************")
    print("checking execution for configuration eight")
       
    belief_cl.set_input_vector([0,0,0,1,1,0])    
    desire_cl.set_input_vector([0,1,1,1,1,0])
    critic_cl.set_input_vector([1,1,1,1,1,1])
    
    IFalcon.planSelectionAndExecution(belief_cl,critic_cl,desire_cl)
    


# In[ ]:





# In[ ]:





# In[ ]:




