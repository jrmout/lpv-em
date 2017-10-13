%TrussClass represents a 2-dimensional truss
%
% See also exempel1, exempel2, exempel3

classdef TrussClass
    
    properties
        
        nodecoords = [];    % List of nodal coordinates     [node#,x-coord,y-xoord]
        elements = [];      % List of element data          [start node#,end node#]
        fixeddofs = [];     % Array containing numbers of the fixed degrees of freedom
        ndof = 0;           % Number of degrees of freedom 
                            % = 2*(number of nodes)-(number of fixed degrees of freedom)
        nel = 0;            % Number of elements
        length = [];        % Length of elements            [nel x 1]
        x = [];             % Element volumes               [nel x 1]
        B = [];             % Geometry dependent matrix     [ndof x nel]
        C = [];             % Geometry dependent matrix     [ndof x nel]
             
        f = [];             % Vector of nodal loads         [ndof x 1]
        c_upp = 0;          % Upper bound on compliance
        
    end
    
    methods
        
        function obj = TrussClass(nodecoords,fixeddofs,maxlength)
            
            % Create a truss object given
            %
            % 1. a list of node coordinates of the form [node number, x-coord, y-coord]
            % 2. a list of fixed degrees of freedom; and
            % 3. a maximum element length  (optional)
            %
            
            if nargin<2
                error('TrussClass requires at least two input arguments.');
            end
            if ~isnumeric(nodecoords) && ~(size(nodecoords,2)==3)
                error('First input to TrussClass should be a numeric array of size n x 3.');
            end
            if ~isnumeric(fixeddofs)
                error('Second argument should be a numeric array');
            end
            if nargin<3
                maxlength = inf;
            elseif ~isnumeric(maxlength) && ~numel(maxlength)==1 && ~maxlength>0
                error('Third input to TrussClass should be a positive scalar');
            end
            
            obj.fixeddofs = fixeddofs;
            obj.nodecoords = nodecoords;
            nd = 2;
            
            % Number of nodes
            nn = size(obj.nodecoords,1);
            
            % Element nodes
            [n1,n2] = meshgrid(1:nn,1:nn);
            % Create elements between each node in the truss
            elem = [n1(:) n2(:)];
            % Remove elements of zero length
            elem(elem(:,1)==elem(:,2),:) = [];
            % Make sure there is only one (and not two) element between 
            % each pair of nodes
            elem = unique(sort(elem,2),'rows');
            % Element lengths = sqrt((x_end-x_start)^2+(y_end-y_start)^2),
            % where x_end and x_start is the x-coord. of the "end" and
            % "start" nodes, respectively, of an element.
            lengths = sqrt((nodecoords(elem(:,2),2)-nodecoords(elem(:,1),2)).^2+...
                (nodecoords(elem(:,2),3)-nodecoords(elem(:,1),3)).^2);
            % Remove elements longer than maxlength
            elem(lengths>maxlength,:) = [];
            lengths(lengths>maxlength,:) = [];
            % Direction cosines = [(x_end-x_start)/length (y_end-y_start)/length]
            cosines = [(nodecoords(elem(:,2),2)-nodecoords(elem(:,1),2))./lengths ...
                (nodecoords(elem(:,2),3)-nodecoords(elem(:,1),3))./lengths];
            
            obj.elements = elem;
            obj.length = lengths;
            obj.nel = size(elem,1);
            
            obj.x = 1*obj.length;
            
            % Set up geometry dependent matrices
            n1 = elem(:,1); n2 = elem(:,2);
            obj.B = sparse(repmat((1:obj.nel)',2*nd,1),[2*n1-1; 2*n1; 2*n2-1; 2*n2],[-cosines(:,1); -cosines(:,2); cosines(:,1);  cosines(:,2)],obj.nel,nd*nn);
            obj.C = sparse(repmat((1:obj.nel)',2*nd,1),[2*n1-1; 2*n1; 2*n2-1; 2*n2],[ cosines(:,2); -cosines(:,1); -cosines(:,2); cosines(:,1)],obj.nel,nd*nn);
            
            % Account for fixed degrees of freedom by removing the
            % correpsonding rows
            obj.B = obj.B(:,setdiff(1:nd*nn,fixeddofs));
            obj.C = obj.C(:,setdiff(1:nd*nn,fixeddofs));
            
            % Number of degrees of freedom
            obj.ndof = size(obj.B,2);                                    
            
        end
        
        
        function draw(obj)
            % Visualize the truss
            linescale = 11;
            minx = 1e-7;
            
            hold on
            for e = 1:obj.nel
                if obj.x(e)>minx*max(obj.x)
                    plot([obj.nodecoords(obj.elements(e,1),2) obj.nodecoords(obj.elements(e,2),2)],...
                         [obj.nodecoords(obj.elements(e,1),3) obj.nodecoords(obj.elements(e,2),3)],...
                         'linewidth',linescale*(obj.x(e)/obj.length(e))/max(obj.x));
                end
            end
            
        end
        
        
    end
    
end
